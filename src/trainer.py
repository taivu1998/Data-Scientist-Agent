import os
import json
import logging
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Any

from src.dataset import AnalysisTaskDataset
from src.model import AnalystAgent
from src.sandbox import SandboxWrapper
from src.utils import setup_logger
from src.validation import OutputValidator

logger = logging.getLogger(__name__)


class Trainer:
    """
    Manages the rigorous evaluation of the agent against the Golden Set.

    Implements the evaluation framework described in the research methodology:
    - Pass@1: Success rate on first attempt (no retries)
    - Pass@3: Success rate within 3 attempts (with visual critic feedback)
    - Execution Success Rate: % of code that runs without exceptions

    This is the 'Evaluation Loop' that produces metrics for the Technical Report.
    """

    def __init__(self, config: dict, dataset: AnalysisTaskDataset):
        self.config = config
        self.dataset = dataset
        self.logger = setup_logger(config["logging"]["log_dir"], "benchmark_run")
        self.agent = AnalystAgent(config)
        self.validator = OutputValidator()

        # Ensure required directories exist
        self._setup_directories()

        # Placeholder for WandB initialization
        # import wandb
        # wandb.init(project=config['experiment_name'])

    def _setup_directories(self):
        """Create necessary directories for data and logs."""
        directories = [self.config["logging"]["log_dir"], self.config["data"]["csv_dir"]]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Ensured directory exists: {directory}")

    def run(self) -> Dict[str, Any]:
        """
        Runs the benchmark loop against the Golden Set.

        Returns:
            Dict containing all evaluation metrics
        """
        results: List[Dict] = []

        # Initialize comprehensive metrics with breakdown by difficulty and type
        metrics = {
            "total": 0,
            "pass_at_1": 0,
            "pass_at_3": 0,
            "pass_refined": 0,
            "execution_success": 0,
            "failures": 0,
            # Per-difficulty metrics
            "by_difficulty": {
                "easy": {
                    "total": 0,
                    "pass_at_1": 0,
                    "pass_at_3": 0,
                    "pass_refined": 0,
                    "execution_success": 0,
                    "failures": 0,
                },
                "medium": {
                    "total": 0,
                    "pass_at_1": 0,
                    "pass_at_3": 0,
                    "pass_refined": 0,
                    "execution_success": 0,
                    "failures": 0,
                },
                "hard": {
                    "total": 0,
                    "pass_at_1": 0,
                    "pass_at_3": 0,
                    "pass_refined": 0,
                    "execution_success": 0,
                    "failures": 0,
                },
            },
            # Per-type metrics
            "by_type": {
                "text": {
                    "total": 0,
                    "pass_at_1": 0,
                    "pass_at_3": 0,
                    "pass_refined": 0,
                    "execution_success": 0,
                    "failures": 0,
                },
                "plot": {
                    "total": 0,
                    "pass_at_1": 0,
                    "pass_at_3": 0,
                    "pass_refined": 0,
                    "execution_success": 0,
                    "failures": 0,
                },
            },
        }

        self.logger.info("=" * 50)
        self.logger.info("Starting Benchmark Evaluation")
        self.logger.info(f"Dataset size: {len(self.dataset)} tasks")
        self.logger.info(
            f"Visual Critic: {'Enabled' if self.config['agent']['enable_visual_critic'] else 'Disabled'}"
        )
        self.logger.info("=" * 50)

        with SandboxWrapper(
            template=self.config["sandbox"]["template"], timeout=self.config["sandbox"]["timeout"]
        ) as sandbox:
            for task in tqdm(self.dataset, desc="Evaluating"):
                metrics["total"] += 1

                # Track by difficulty
                difficulty = task.get("difficulty", "unknown")
                if difficulty in metrics["by_difficulty"]:
                    metrics["by_difficulty"][difficulty]["total"] += 1

                # Track by type
                task_type = task.get("type", "unknown")
                if task_type in metrics["by_type"]:
                    metrics["by_type"][task_type]["total"] += 1

                task_result = self._evaluate_task(task, sandbox, metrics)
                results.append(task_result)

        # Calculate final metrics
        metrics = self._calculate_final_metrics(metrics)

        self._save_results(results, metrics)
        self._log_summary(metrics)

        return metrics

    def _evaluate_task(self, task: Dict, sandbox: SandboxWrapper, metrics: Dict) -> Dict:
        """
        Evaluates a single task from the Golden Set.

        Args:
            task: Task dictionary with id, query, csv_name, type
            sandbox: Active sandbox wrapper
            metrics: Metrics dict to update

        Returns:
            Result dictionary for this task
        """
        task_id = task["id"]
        difficulty = task.get("difficulty", "unknown")
        task_type = task.get("type", "unknown")

        self.logger.info(f"Evaluating task: {task_id}")

        # Setup data file
        local_csv = os.path.join(self.config["data"]["csv_dir"], task["csv_name"])

        # Create dummy CSV if missing (for reproducibility)
        if not os.path.exists(local_csv):
            self._create_dummy_csv(local_csv)

        # Upload to sandbox
        try:
            sandbox.upload_data(local_csv, "data.csv")
        except Exception as e:
            self.logger.error(f"Failed to upload data for task {task_id}: {e}")
            metrics["failures"] += 1
            self._update_granular_metrics(
                metrics, difficulty, task_type, success=False, has_error=True
            )
            return {"id": task_id, "success": False, "error": str(e)}

        # Extract semantic context (The Research Twist - Semantic Compression)
        context_str = self.dataset.get_semantic_context(local_csv)

        try:
            # Run the agent
            final_state = self.agent.run(task["query"], context_str, sandbox)

            # Extract metrics
            retry_count = final_state.get("retry_count", 0)
            exec_result = final_state.get("execution_result", {})
            is_solved = final_state.get("is_solved", False)
            generated_code = final_state.get("generated_code", "")
            visual_feedback = exec_result.get("visual_feedback", "")

            # Check for execution errors
            has_error = bool(exec_result.get("error") or exec_result.get("stderr"))

            # Validate output against golden set expectations
            is_valid_output, validation_feedback = self.validator.validate_task(
                task, exec_result, generated_code
            )

            # Combine: must have no errors, pass visual critic, and pass output validation
            is_success = not has_error and is_solved and is_valid_output

            # Update overall metrics
            if not has_error:
                metrics["execution_success"] += 1

            if is_success:
                # Pass@1: Success on first attempt
                if retry_count == 1:
                    metrics["pass_at_1"] += 1

                # Pass@3: Success within 3 attempts
                if retry_count <= 3:
                    metrics["pass_at_3"] += 1

                # Pass refined: Any successful completion
                metrics["pass_refined"] += 1
            else:
                metrics["failures"] += 1

            # Update granular metrics
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
                "is_valid_output": is_valid_output,
                "validation_feedback": validation_feedback,
                "stdout": exec_result.get("stdout", ""),
                "stderr": exec_result.get("stderr", ""),
                "visual_feedback": visual_feedback,
                "generated_code": generated_code,
            }

            self.logger.info(
                f"Task {task_id}: {'SUCCESS' if is_success else 'FAILED'} (retries: {retry_count}) - {validation_feedback}"
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
        """Creates a dummy CSV for reproducibility if the real file is missing.

        Uses columns from actual salaries.csv: Job Title, Salary, Experience, Department, Location, Education
        """
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
        """Update granular metrics by difficulty and type."""
        # Update by difficulty
        if difficulty in metrics["by_difficulty"]:
            m = metrics["by_difficulty"][difficulty]
            if not has_error:
                m["execution_success"] += 1
            if success:
                if retry_count == 1:
                    m["pass_at_1"] += 1
                if retry_count <= 3:
                    m["pass_at_3"] += 1
                m["pass_refined"] += 1
            else:
                m["failures"] += 1

        # Update by type
        if task_type in metrics["by_type"]:
            m = metrics["by_type"][task_type]
            if not has_error:
                m["execution_success"] += 1
            if success:
                if retry_count == 1:
                    m["pass_at_1"] += 1
                if retry_count <= 3:
                    m["pass_at_3"] += 1
                m["pass_refined"] += 1
            else:
                m["failures"] += 1

    def _calculate_final_metrics(self, metrics: Dict) -> Dict:
        """Calculate percentage-based final metrics including granular breakdowns."""
        total = metrics["total"]
        if total == 0:
            return metrics

        # Overall rates
        metrics["pass_at_1_rate"] = round(metrics["pass_at_1"] / total * 100, 2)
        metrics["pass_at_3_rate"] = round(metrics["pass_at_3"] / total * 100, 2)
        metrics["pass_refined_rate"] = round(metrics["pass_refined"] / total * 100, 2)
        metrics["execution_success_rate"] = round(metrics["execution_success"] / total * 100, 2)
        metrics["failure_rate"] = round(metrics["failures"] / total * 100, 2)

        # Per-difficulty rates
        for difficulty in ["easy", "medium", "hard"]:
            if difficulty in metrics["by_difficulty"]:
                m = metrics["by_difficulty"][difficulty]
                if m["total"] > 0:
                    m["pass_at_1_rate"] = round(m["pass_at_1"] / m["total"] * 100, 2)
                    m["pass_at_3_rate"] = round(m["pass_at_3"] / m["total"] * 100, 2)
                    m["pass_refined_rate"] = round(m["pass_refined"] / m["total"] * 100, 2)
                    m["execution_success_rate"] = round(
                        m["execution_success"] / m["total"] * 100, 2
                    )
                    m["failure_rate"] = round(m["failures"] / m["total"] * 100, 2)

        # Per-type rates
        for task_type in ["text", "plot"]:
            if task_type in metrics["by_type"]:
                m = metrics["by_type"][task_type]
                if m["total"] > 0:
                    m["pass_at_1_rate"] = round(m["pass_at_1"] / m["total"] * 100, 2)
                    m["pass_at_3_rate"] = round(m["pass_at_3"] / m["total"] * 100, 2)
                    m["pass_refined_rate"] = round(m["pass_refined"] / m["total"] * 100, 2)
                    m["execution_success_rate"] = round(
                        m["execution_success"] / m["total"] * 100, 2
                    )
                    m["failure_rate"] = round(m["failures"] / m["total"] * 100, 2)

        return metrics

    def _save_results(self, results: List[Dict], metrics: Dict):
        """Save detailed results and metrics to JSON."""
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
        """Log a summary of the evaluation results with granular breakdown."""
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
            f"Execution Success: {metrics['execution_success']} ({metrics.get('execution_success_rate', 0)}%)"
        )
        self.logger.info(f"Failures: {metrics['failures']} ({metrics.get('failure_rate', 0)}%)")

        # Per-difficulty breakdown
        self.logger.info("-" * 60)
        self.logger.info("BY DIFFICULTY:")
        for difficulty in ["easy", "medium", "hard"]:
            if difficulty in metrics.get("by_difficulty", {}):
                m = metrics["by_difficulty"][difficulty]
                self.logger.info(
                    f"  {difficulty.upper()}: {m['total']} tasks, "
                    f"Pass@1: {m.get('pass_at_1_rate', 0)}%, "
                    f"Pass@3: {m.get('pass_at_3_rate', 0)}%"
                )

        # Per-type breakdown
        self.logger.info("-" * 60)
        self.logger.info("BY TYPE:")
        for task_type in ["text", "plot"]:
            if task_type in metrics.get("by_type", {}):
                m = metrics["by_type"][task_type]
                self.logger.info(
                    f"  {task_type.upper()}: {m['total']} tasks, "
                    f"Pass@1: {m.get('pass_at_1_rate', 0)}%, "
                    f"Pass@3: {m.get('pass_at_3_rate', 0)}%"
                )

        self.logger.info("=" * 60)
