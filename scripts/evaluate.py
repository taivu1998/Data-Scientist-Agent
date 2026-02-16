#!/usr/bin/env python3
"""
Benchmark evaluation script for the Auto-Analyst agent.

Usage:
    python scripts/evaluate.py --config configs/default.yaml

This script runs the full benchmark evaluation against the Golden Set,
producing metrics like Pass@1, Pass@3, and Execution Success Rate.
"""
import sys
import os
import logging

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_parser import parse_args
from src.utils import seed_everything
from src.dataset import AnalysisTaskDataset
from src.trainer import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run the benchmark evaluation."""
    config = parse_args()

    # Set random seed for reproducibility
    seed_everything(config['seed'])

    print("=" * 60)
    print("AUTO-ANALYST BENCHMARK EVALUATION")
    print("=" * 60)
    print(f"Experiment: {config.get('experiment_name', 'unnamed')}")
    print(f"Model: {config['agent']['model_id']}")
    print(f"Visual Critic: {'Enabled' if config['agent']['enable_visual_critic'] else 'Disabled'}")
    print(f"Max Retries: {config['agent']['max_retries']}")
    print(f"Benchmark: {config['data']['benchmark_path']}")
    print("=" * 60)

    # Load the Golden Set
    dataset = AnalysisTaskDataset(config['data']['benchmark_path'])
    print(f"\nLoaded {len(dataset)} tasks from Golden Set")

    # Show task breakdown
    easy_count = len(dataset.get_tasks_by_difficulty('easy'))
    medium_count = len(dataset.get_tasks_by_difficulty('medium'))
    hard_count = len(dataset.get_tasks_by_difficulty('hard'))
    print(f"  - Easy: {easy_count}")
    print(f"  - Medium: {medium_count}")
    print(f"  - Hard: {hard_count}")

    # Run evaluation
    trainer = Trainer(config, dataset)

    try:
        metrics = trainer.run()

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Total Tasks:        {metrics['total']}")
        print(f"Pass@1:             {metrics['pass_at_1']} ({metrics.get('pass_at_1_rate', 0)}%)")
        print(f"Pass@3:             {metrics['pass_at_3']} ({metrics.get('pass_at_3_rate', 0)}%)")
        print(f"Pass (Refined):     {metrics['pass_refined']} ({metrics.get('pass_refined_rate', 0)}%)")
        print(f"Execution Success:  {metrics['execution_success']} ({metrics.get('execution_success_rate', 0)}%)")
        print(f"Failures:           {metrics['failures']} ({metrics.get('failure_rate', 0)}%)")
        print("=" * 60)

        # Highlight the key research metric
        improvement = metrics.get('pass_at_3_rate', 0) - metrics.get('pass_at_1_rate', 0)
        print(f"\nVisual Critic Improvement: +{improvement:.1f}% (Pass@3 - Pass@1)")

        print(f"\nDetailed results saved to: {config['logging']['log_dir']}/results.json")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
