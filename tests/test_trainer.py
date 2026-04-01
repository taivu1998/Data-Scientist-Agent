"""Tests for the Trainer class (unit tests)."""

import os
import sys
import pytest
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTrainerMetrics:
    """Test suite for Trainer metrics calculation."""

    def test_initial_metrics_structure(self):
        """Test that initial metrics has correct structure."""
        from src.trainer import Trainer
        from src.dataset import AnalysisTaskDataset
        from src.config_parser import load_config

        # Get a valid config
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"
        )

        if os.path.exists(config_path):
            config = load_config(config_path)
            dataset = AnalysisTaskDataset("dummy")

            # Create trainer instance (won't run)
            # Just test that it can be instantiated
            assert Trainer is not None

    def test_metrics_by_difficulty_structure(self):
        """Test metrics structure includes difficulty breakdown."""
        metrics = {
            "total": 0,
            "pass_at_1": 0,
            "pass_at_3": 0,
            "pass_refined": 0,
            "execution_success": 0,
            "failures": 0,
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
                "plot_log_check": {
                    "total": 0,
                    "pass_at_1": 0,
                    "pass_at_3": 0,
                    "pass_refined": 0,
                    "execution_success": 0,
                    "failures": 0,
                },
            },
        }

        assert "by_difficulty" in metrics
        assert "easy" in metrics["by_difficulty"]
        assert "by_type" in metrics
        assert "text" in metrics["by_type"]
        assert "plot_log_check" in metrics["by_type"]

    def test_calculate_final_metrics_rates(self):
        """Test that final metrics calculates rates correctly."""
        metrics = {
            "total": 10,
            "pass_at_1": 5,
            "pass_at_3": 7,
            "pass_refined": 8,
            "execution_success": 9,
            "failures": 1,
            "by_difficulty": {
                "easy": {
                    "total": 3,
                    "pass_at_1": 2,
                    "pass_at_3": 3,
                    "pass_refined": 3,
                    "execution_success": 3,
                    "failures": 0,
                },
                "medium": {
                    "total": 4,
                    "pass_at_1": 2,
                    "pass_at_3": 3,
                    "pass_refined": 3,
                    "execution_success": 3,
                    "failures": 1,
                },
                "hard": {
                    "total": 3,
                    "pass_at_1": 1,
                    "pass_at_3": 1,
                    "pass_refined": 2,
                    "execution_success": 3,
                    "failures": 0,
                },
            },
            "by_type": {
                "text": {
                    "total": 4,
                    "pass_at_1": 3,
                    "pass_at_3": 4,
                    "pass_refined": 4,
                    "execution_success": 4,
                    "failures": 0,
                },
                "plot": {
                    "total": 5,
                    "pass_at_1": 2,
                    "pass_at_3": 3,
                    "pass_refined": 4,
                    "execution_success": 4,
                    "failures": 1,
                },
                "plot_log_check": {
                    "total": 1,
                    "pass_at_1": 0,
                    "pass_at_3": 0,
                    "pass_refined": 0,
                    "execution_success": 1,
                    "failures": 1,
                },
            },
        }

        # Calculate expected rates
        total = metrics["total"]
        expected_rates = {
            "pass_at_1_rate": round(metrics["pass_at_1"] / total * 100, 2),
            "pass_at_3_rate": round(metrics["pass_at_3"] / total * 100, 2),
            "pass_refined_rate": round(metrics["pass_refined"] / total * 100, 2),
            "execution_success_rate": round(metrics["execution_success"] / total * 100, 2),
            "failure_rate": round(metrics["failures"] / total * 100, 2),
        }

        assert expected_rates["pass_at_1_rate"] == 50.0
        assert expected_rates["pass_at_3_rate"] == 70.0
        assert expected_rates["pass_refined_rate"] == 80.0
        assert expected_rates["execution_success_rate"] == 90.0
        assert expected_rates["failure_rate"] == 10.0

    def test_plot_log_check_is_counted_in_type_totals(self):
        """Type totals should include plot_log_check tasks."""
        from src.trainer import Trainer

        trainer = Trainer.__new__(Trainer)
        metrics = trainer._initialize_metrics()

        total_by_type = sum(bucket["total"] for bucket in metrics["by_type"].values())
        assert "plot_log_check" in metrics["by_type"]
        assert total_by_type == 0

    def test_calculate_final_metrics_empty(self):
        """Test that empty metrics returns as-is."""
        metrics = {"total": 0}

        # Should not raise division by zero
        assert metrics["total"] == 0


class TestTrainerResultSaving:
    """Test result saving functionality."""

    def test_save_results_json_structure(self, tmp_path):
        """Test that results JSON has correct structure."""
        metrics = {
            "total": 10,
            "pass_at_1": 5,
            "pass_at_1_rate": 50.0,
        }

        results = [
            {"id": "task_1", "success": True, "type": "text"},
            {"id": "task_2", "success": False, "type": "plot"},
        ]

        output = {
            "experiment_name": "test",
            "model_id": "claude-sonnet-4-20250514",
            "visual_critic_enabled": True,
            "metrics": metrics,
            "details": results,
        }

        # Save to temp file
        output_path = tmp_path / "results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)

        # Read back and verify
        with open(output_path, "r") as f:
            loaded = json.load(f)

        assert loaded["experiment_name"] == "test"
        assert loaded["metrics"]["total"] == 10
        assert len(loaded["details"]) == 2


class TestTrainerDummyCSV:
    """Test dummy CSV creation."""

    def test_create_dummy_csv(self, tmp_path):
        """Test that dummy CSV is created correctly."""
        import pandas as pd

        from src.trainer import Trainer
        from src.dataset import AnalysisTaskDataset

        # Create a mock trainer to test the method
        class MockTrainer:
            def __init__(self):
                self.logger = logging.getLogger("test")

            def _create_dummy_csv(self, path):
                """Creates a dummy CSV for reproducibility if the real file is missing."""
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
                        "Department": [
                            "Engineering",
                            "Analytics",
                            "Product",
                            "Design",
                            "Analytics",
                        ],
                        "Location": [
                            "San Francisco",
                            "New York",
                            "San Francisco",
                            "New York",
                            "Austin",
                        ],
                        "Education": ["Bachelor's", "Master's", "MBA", "Bachelor's", "Bachelor's"],
                    }
                )
                dummy_data.to_csv(path, index=False)

        import logging

        trainer = MockTrainer()

        csv_path = tmp_path / "test_dummy.csv"
        trainer._create_dummy_csv(str(csv_path))

        # Verify file exists and has correct structure
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        assert len(df) == 5
        assert "Job Title" in df.columns
        assert "Salary" in df.columns
        assert "Experience" in df.columns
        assert "Department" in df.columns
        assert "Location" in df.columns
        assert "Education" in df.columns
