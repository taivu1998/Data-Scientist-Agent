#!/usr/bin/env python3
"""
Single-query inference script for the Auto-Analyst agent.

Usage:
    python scripts/inference.py --config configs/default.yaml --query "What is the average salary?" --csv_path data/salaries.csv

This script allows running the agent on a single query for testing and demonstration.
"""
import sys
import os
import base64
import logging

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_parser import parse_args
from src.model import AnalystAgent
from src.sandbox import SandboxWrapper
from src.dataset import AnalysisTaskDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_image(base64_data: str, output_path: str = "output.png"):
    """Save base64 encoded image to file."""
    try:
        image_bytes = base64.b64decode(base64_data)
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
        logger.info(f"Image saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return None


def main():
    """Run a single inference with the Auto-Analyst agent."""
    config = parse_args()

    query = config['runtime'].get('query')
    csv_path = config['runtime'].get('csv_path')

    if not query or not csv_path:
        print("Error: Both --query and --csv_path are required")
        print("\nUsage:")
        print("  python scripts/inference.py --config configs/default.yaml \\")
        print('    --query "What is the average salary?" \\')
        print("    --csv_path data/salaries.csv")
        sys.exit(1)

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    # Profile the dataset using Semantic Compressor
    ds = AnalysisTaskDataset("dummy")  # Path not needed for context extraction
    context_str = ds.get_semantic_context(csv_path)

    print("=" * 60)
    print("AUTO-ANALYST INFERENCE")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Data:  {csv_path}")
    print(f"Model: {config['agent']['model_id']}")
    print(f"Visual Critic: {'Enabled' if config['agent']['enable_visual_critic'] else 'Disabled'}")
    print("=" * 60)

    print("\nDataset Context:")
    print("-" * 40)
    print(context_str)
    print("-" * 40)

    print("\nRunning agent...")

    try:
        with SandboxWrapper(
            template=config['sandbox']['template'],
            timeout=config['sandbox']['timeout']
        ) as sandbox:
            # Upload data to sandbox
            sandbox.upload_data(csv_path, "data.csv")

            # Create and run agent
            agent = AnalystAgent(config)
            final_state = agent.run(query, context_str, sandbox)

            # Extract results
            exec_result = final_state.get('execution_result', {})
            is_solved = final_state.get('is_solved', False)
            retry_count = final_state.get('retry_count', 0)

            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)

            # Status
            status = "SUCCESS" if is_solved and not exec_result.get('error') else "FAILED"
            print(f"Status: {status}")
            print(f"Attempts: {retry_count}")

            # Generated code
            code = final_state.get('generated_code', '')
            if code:
                print("\nGenerated Code:")
                print("-" * 40)
                print(code)
                print("-" * 40)

            # Standard output
            if exec_result.get('stdout'):
                print("\nOutput:")
                print("-" * 40)
                print(exec_result['stdout'])
                print("-" * 40)

            # Errors
            if exec_result.get('stderr'):
                print("\nStderr:")
                print("-" * 40)
                print(exec_result['stderr'])
                print("-" * 40)

            if exec_result.get('error'):
                print(f"\nError: {exec_result['error']}")

            # Visual feedback
            if exec_result.get('visual_feedback'):
                print("\nVisual Critic Feedback:")
                print("-" * 40)
                print(exec_result['visual_feedback'])
                print("-" * 40)

            # Image output
            if exec_result.get('image_base64'):
                output_path = save_image(exec_result['image_base64'])
                if output_path:
                    print(f"\nPlot saved to: {output_path}")

            print("=" * 60)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
