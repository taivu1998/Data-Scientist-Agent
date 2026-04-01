import json
import logging
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.dataset import AnalysisTaskDataset
from src.model import AnalystAgent
from src.sandbox import SandboxWrapper
from src.utils import setup_logger
from src.validation import OutputValidator

logger = logging.getLogger(__name__)


class Trainer:
    """
    Manage benchmark evaluation of the agent against the Golden Set.
    """

    def __init__(self, config: dict, dataset: AnalysisTaskDataset):
        self.config = config
        self.dataset = dataset
        self.logger = setup_logger(config["logging"]["log_dir"], "benchmark_run")
        self.agent: Optional[AnalystAgent] = None
        self.validator = OutputValidator()
        self._setup_directories()

    def _setup_directories(self):
        directories = [self.config["logging"]["log_dir"], self.config["data"]["csv_dir"]]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Ensured directory exists: {directory}")

    def run(self) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        metrics = self._initialize_metrics()

        self.logger.info("=" * 50)
        self.logger.info("Starting Benchmark Evaluation")
        self.logger.info(f"Dataset size: {len(self.dataset)} tasks")
        self.logger.info(
            f"Visual Critic: {'Enabled' if self.config['agent']['enable_visual_critic'] else 'Disabled'}"
        )
        self.logger.info("=" * 50)

        self.agent = self.agent or AnalystAgent(self.config)

        for task in tqdm(self.dataset, desc="Evaluating"):
            metrics["total"] += 1
            difficulty = task.get("difficulty", "unknown")
            task_type = task.get("type", "unknown")

            if difficulty in metrics["by_difficulty"]:
                metrics["by_difficulty"][difficulty]["total"] += 1
            if task_type not in metrics["by_type"]:
                metrics["by_type"][task_type] = self._blank_metric_bucket()
            metrics["by_type"][task_type]["total"] += 1

            with SandboxWrapper(
                template=self.config["sandbox"]["template"],
                timeout=self.config["sandbox"]["timeout"],
            ) as sandbox:
                task_result = self._evaluate_task(task, sandbox, metrics)
            results.append(task_result)

        metrics = self._calculate_final_metrics(metrics)
        self._save_results(results, metrics)
        self._log_summary(metrics)
        return metrics

    def _initialize_metrics(self) -> Dict[str, Any]:
        return {
            "total": 0,
            "pass_at_1": 0,
            "pass_at_3": 0,
            "pass_refined": 0,
            "execution_success": 0,
            "failures": 0,
            "by_difficulty": {
                "easy": self._blank_metric_bucket(),
                "medium": self._blank_metric_bucket(),
                "hard": self._blank_metric_bucket(),
            },
            "by_type": {
                "text": self._blank_metric_bucket(),
                "plot": self._blank_metric_bucket(),
                "plot_log_check": self._blank_metric_bucket(),
            },
        }

    def _blank_metric_bucket(self) -> Dict[str, int]:
        return {
            "total": 0,
            "pass_at_1": 0,
            "pass_at_3": 0,
            "pass_refined": 0,
            "execution_success": 0,
            "failures": 0,
        }

    def _evaluate_task(self, task: Dict, sandbox: SandboxWrapper, metrics: Dict) -> Dict:
        task_id = task["id"]
        difficulty = task.get("difficulty", "unknown")
        task_type = task.get("type", "unknown")
        self.logger.info(f"Evaluating task: {task_id}")

        local_csv = os.path.join(self.config["data"]["csv_dir"], task["csv_name"])
        if not os.path.exists(local_csv):
            self._create_dummy_csv(local_csv)

        try:
            sandbox.upload_data(local_csv, "data.csv")
        except Exception as e:
            self.logger.error(f"Failed to upload data for task {task_id}: {e}")
            metrics["failures"] += 1
            self._update_granular_metrics(
                metrics, difficulty, task_type, success=False, has_error=True
            )
            return {"id": task_id, "success": False, "error": str(e)}

        context_str = self.dataset.get_semantic_context(local_csv)

        try:
            final_state = self.agent.run(
                task["query"],
                context_str,
                sandbox,
                task_type=task_type,
                critic_failure_mode="strict",
            )
            retry_count = final_state.get("retry_count", 0)
            exec_result = final_state.get("execution_result", {})
            is_solved = final_state.get("is_solved", False)
            generated_code = final_state.get("generated_code", "")

            has_error = exec_result.get("status") == "error" or bool(exec_result.get("error"))
            validation = self.validator.validate_task_result(
                task, exec_result, generated_code, is_solved
            )
            is_success = validation.passed

            if not has_error:
                metrics["execution_success"] += 1

            if is_success:
                if retry_count == 1:
                    metrics["pass_at_1"] += 1
                if retry_count <= 3:
                    metrics["pass_at_3"] += 1
                metrics["pass_refined"] += 1
            else:
                metrics["failures"] += 1

            self._update_granular_metrics(
                metrics,
                difficulty,
                task_type,
                success=is_success,
                has_error=has_error,
                retry_count=retry_count,
            )

            result = {
                "id": task_id,
                "query": task["query"],
                "type": task_type,
                "difficulty": difficulty,
                "success": is_success,
                "retry_count": retry_count,
                "is_solved": is_solved,
                "has_error": has_error,
                "validation": asdict(validation),
                "validation_feedback": validation.summary,
                "stdout": exec_result.get("stdout", ""),
                "stderr": exec_result.get("stderr", ""),
                "warnings": exec_result.get("warnings", []),
                "visual_feedback": exec_result.get("visual_feedback", ""),
                "critic_status": exec_result.get("critic_status", ""),
                "generated_code": generated_code,
            }

            self.logger.info(
                f"Task {task_id}: {'SUCCESS' if is_success else 'FAILED'} "
                f"(retries: {retry_count}) - {validation.summary}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Task {task_id} failed with exception: {e}")
            metrics["failures"] += 1
            self._update_granular_metrics(
                metrics, difficulty, task_type, success=False, has_error=True
            )
            return {"id": task_id, "query": task["query"], "success": False, "error": str(e)}

    def _create_dummy_csv(self, path: str):
        dummy_data = pd.DataFrame(
            {
                "Job Title": [
                    "Software Engineer",
                    "Data Scientist",
                    "Product Manager",
                    "Designer",
                    "Analyst",
                ],
                "Salary": [95000, 105000, 110000, 85000, 75000],
                "Experience": [3, 4, 5, 3, 2],
                "Department": ["Engineering", "Analytics", "Product", "Design", "Analytics"],
                "Location": ["San Francisco", "New York", "San Francisco", "New York", "Austin"],
                "Education": ["Bachelor's", "Master's", "MBA", "Bachelor's", "Bachelor's"],
            }
        )
        dummy_data.to_csv(path, index=False)
        self.logger.info(f"Created dummy CSV: {path}")

    def _update_granular_metrics(
        self,
        metrics: Dict,
        difficulty: str,
        task_type: str,
        success: bool = False,
        has_error: bool = False,
        retry_count: int = 0,
    ):
        if difficulty in metrics["by_difficulty"]:
            bucket = metrics["by_difficulty"][difficulty]
            self._update_metric_bucket(bucket, success, has_error, retry_count)

        if task_type not in metrics["by_type"]:
            metrics["by_type"][task_type] = self._blank_metric_bucket()
        self._update_metric_bucket(metrics["by_type"][task_type], success, has_error, retry_count)

    def _update_metric_bucket(
        self, bucket: Dict[str, int], success: bool, has_error: bool, retry_count: int
    ):
        if not has_error:
            bucket["execution_success"] += 1
        if success:
            if retry_count == 1:
                bucket["pass_at_1"] += 1
            if retry_count <= 3:
                bucket["pass_at_3"] += 1
            bucket["pass_refined"] += 1
        else:
            bucket["failures"] += 1

    def _calculate_final_metrics(self, metrics: Dict) -> Dict:
        total = metrics["total"]
        if total == 0:
            return metrics

        metrics["pass_at_1_rate"] = round(metrics["pass_at_1"] / total * 100, 2)
        metrics["pass_at_3_rate"] = round(metrics["pass_at_3"] / total * 100, 2)
        metrics["pass_refined_rate"] = round(metrics["pass_refined"] / total * 100, 2)
        metrics["execution_success_rate"] = round(metrics["execution_success"] / total * 100, 2)
        metrics["failure_rate"] = round(metrics["failures"] / total * 100, 2)

        for bucket in metrics["by_difficulty"].values():
            self._calculate_bucket_rates(bucket)
        for bucket in metrics["by_type"].values():
            self._calculate_bucket_rates(bucket)

        return metrics

    def _calculate_bucket_rates(self, bucket: Dict[str, Any]):
        if bucket["total"] <= 0:
            return

        bucket["pass_at_1_rate"] = round(bucket["pass_at_1"] / bucket["total"] * 100, 2)
        bucket["pass_at_3_rate"] = round(bucket["pass_at_3"] / bucket["total"] * 100, 2)
        bucket["pass_refined_rate"] = round(bucket["pass_refined"] / bucket["total"] * 100, 2)
        bucket["execution_success_rate"] = round(
            bucket["execution_success"] / bucket["total"] * 100, 2
        )
        bucket["failure_rate"] = round(bucket["failures"] / bucket["total"] * 100, 2)

    def _save_results(self, results: List[Dict], metrics: Dict):
        path = os.path.join(self.config["logging"]["log_dir"], "results.json")
        output = {
            "experiment_name": self.config.get("experiment_name", "unnamed"),
            "model_id": self.config["agent"]["model_id"],
            "visual_critic_enabled": self.config["agent"]["enable_visual_critic"],
            "metrics": metrics,
            "details": results,
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=4)

        self.logger.info(f"Results saved to {path}")

    def _log_summary(self, metrics: Dict):
        self.logger.info("=" * 60)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Tasks: {metrics['total']}")
        self.logger.info(f"Pass@1: {metrics['pass_at_1']} ({metrics.get('pass_at_1_rate', 0)}%)")
        self.logger.info(f"Pass@3: {metrics['pass_at_3']} ({metrics.get('pass_at_3_rate', 0)}%)")
        self.logger.info(
            f"Pass (Refined): {metrics['pass_refined']} ({metrics.get('pass_refined_rate', 0)}%)"
        )
        self.logger.info(
            f"Execution Success: {metrics['execution_success']} "
            f"({metrics.get('execution_success_rate', 0)}%)"
        )
        self.logger.info(f"Failures: {metrics['failures']} ({metrics.get('failure_rate', 0)}%)")

        self.logger.info("-" * 60)
        self.logger.info("BY DIFFICULTY:")
        for difficulty in ["easy", "medium", "hard"]:
            if difficulty in metrics.get("by_difficulty", {}):
                bucket = metrics["by_difficulty"][difficulty]
                self.logger.info(
                    f"  {difficulty.upper()}: {bucket['total']} tasks, "
                    f"Pass@1: {bucket.get('pass_at_1_rate', 0)}%, "
                    f"Pass@3: {bucket.get('pass_at_3_rate', 0)}%"
                )

        self.logger.info("-" * 60)
        self.logger.info("BY TYPE:")
        for task_type in ["text", "plot", "plot_log_check"]:
            if task_type in metrics.get("by_type", {}):
                bucket = metrics["by_type"][task_type]
                self.logger.info(
                    f"  {task_type.upper()}: {bucket['total']} tasks, "
                    f"Pass@1: {bucket.get('pass_at_1_rate', 0)}%, "
                    f"Pass@3: {bucket.get('pass_at_3_rate', 0)}%"
                )

        self.logger.info("=" * 60)
