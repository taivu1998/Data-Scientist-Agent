"""Tests for the OutputValidator class."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validation import OutputValidator, validate_task_output


class TestOutputValidator:
    """Test suite for OutputValidator."""

    def test_validate_text_task_success(self):
        """Should pass when expected output is in stdout."""
        validator = OutputValidator()

        task = {"id": "test_1", "type": "text", "expected_output_contains": ["25", "6"]}

        exec_result = {
            "stdout": "Data has 25 rows and 6 columns",
            "stderr": "",
            "error": None,
            "image_base64": None,
        }

        is_valid, feedback = validator.validate_task(task, exec_result, "")

        assert is_valid is True
        assert "validated" in feedback.lower()

    def test_validate_text_task_failure(self):
        """Should fail when expected output is not in stdout."""
        validator = OutputValidator()

        task = {"id": "test_1", "type": "text", "expected_output_contains": ["1000", "rows"]}

        exec_result = {
            "stdout": "Data has 25 rows and 6 columns",
            "stderr": "",
            "error": None,
            "image_base64": None,
        }

        is_valid, feedback = validator.validate_task(task, exec_result, "")

        assert is_valid is False
        assert "not found" in feedback.lower()

    def test_validate_text_task_no_expected(self):
        """Should pass when no expected output specified."""
        validator = OutputValidator()

        task = {"id": "test_1", "type": "text"}

        exec_result = {"stdout": "Some output", "stderr": "", "error": None}

        is_valid, feedback = validator.validate_task(task, exec_result, "")

        assert is_valid is True

    def test_validate_plot_no_image(self):
        """Should fail when no image generated for plot task."""
        validator = OutputValidator()

        task = {"id": "test_1", "type": "plot", "expected_visual": "bar_chart"}

        exec_result = {"stdout": "", "stderr": "", "error": None, "image_base64": None}

        is_valid, feedback = validator.validate_task(task, exec_result, "")

        assert is_valid is False
        assert "no image" in feedback.lower()

    def test_validate_plot_bar_chart(self):
        """Should validate bar chart plot type."""
        validator = OutputValidator()

        task = {"id": "test_1", "type": "plot", "expected_visual": "bar_chart"}

        code = "plt.bar(department, salary)\nplt.show()"

        exec_result = {
            "stdout": "",
            "stderr": "",
            "error": None,
            "image_base64": "fake_base64_image",
        }

        is_valid, feedback = validator.validate_task(task, exec_result, code)

        assert is_valid is True
        assert "bar" in feedback.lower()

    def test_validate_plot_histogram(self):
        """Should validate histogram plot type."""
        validator = OutputValidator()

        task = {"id": "test_1", "type": "plot", "expected_visual": "histogram"}

        code = "plt.hist(salary, bins=10)\nplt.show()"

        exec_result = {
            "stdout": "",
            "stderr": "",
            "error": None,
            "image_base64": "fake_base64_image",
        }

        is_valid, feedback = validator.validate_task(task, exec_result, code)

        assert is_valid is True

    def test_validate_plot_scatter(self):
        """Should validate scatter plot type."""
        validator = OutputValidator()

        task = {"id": "test_1", "type": "plot", "expected_visual": "scatter"}

        code = "plt.scatter(x, y)\nplt.show()"

        exec_result = {
            "stdout": "",
            "stderr": "",
            "error": None,
            "image_base64": "fake_base64_image",
        }

        is_valid, feedback = validator.validate_task(task, exec_result, code)

        assert is_valid is True

    def test_validate_plot_heatmap(self):
        """Should validate heatmap plot type."""
        validator = OutputValidator()

        task = {"id": "test_1", "type": "plot", "expected_visual": "heatmap"}

        code = "sns.heatmap(df.corr())\nplt.show()"

        exec_result = {
            "stdout": "",
            "stderr": "",
            "error": None,
            "image_base64": "fake_base64_image",
        }

        is_valid, feedback = validator.validate_task(task, exec_result, code)

        assert is_valid is True

    def test_validate_plot_wrong_type(self):
        """Should fail when plot type doesn't match."""
        validator = OutputValidator()

        task = {"id": "test_1", "type": "plot", "expected_visual": "bar_chart"}

        code = "plt.scatter(x, y)\nplt.show()"

        exec_result = {
            "stdout": "",
            "stderr": "",
            "error": None,
            "image_base64": "fake_base64_image",
        }

        is_valid, feedback = validator.validate_task(task, exec_result, code)

        assert is_valid is False

    def test_validate_log_scale(self):
        """Should validate log scale for plot_log_check."""
        validator = OutputValidator()

        task = {"id": "test_1", "type": "plot_log_check", "expected_visual": "scatter_log"}

        code = "plt.yscale('log')\nplt.scatter(x, y)\nplt.show()"

        exec_result = {
            "stdout": "",
            "stderr": "",
            "error": None,
            "image_base64": "fake_base64_image",
        }

        is_valid, feedback = validator.validate_task(task, exec_result, code)

        assert is_valid is True
        assert "log" in feedback.lower()

    def test_validate_log_scale_missing(self):
        """Should fail when log scale is missing."""
        validator = OutputValidator()

        task = {"id": "test_1", "type": "plot_log_check", "expected_visual": "scatter_log"}

        code = "plt.scatter(x, y)\nplt.show()"

        exec_result = {
            "stdout": "",
            "stderr": "",
            "error": None,
            "image_base64": "fake_base64_image",
        }

        is_valid, feedback = validator.validate_task(task, exec_result, code)

        assert is_valid is False
        assert "log scale" in feedback.lower()

    def test_validate_execution_error(self):
        """Should fail on execution error."""
        validator = OutputValidator()

        task = {"id": "test_1", "type": "text"}

        exec_result = {
            "stdout": "",
            "stderr": "NameError: name 'df' is not defined",
            "error": "NameError",
        }

        is_valid, feedback = validator.validate_task(task, exec_result, "")

        assert is_valid is False
        assert "error" in feedback.lower()

    def test_warning_only_stderr_does_not_fail_validation(self):
        """Warning-only stderr should not count as execution failure."""
        validator = OutputValidator()

        task = {"id": "test_1", "type": "text", "expected_output_contains": ["25", "rows"]}
        exec_result = {
            "status": "warning",
            "stdout": "Data has 25 rows",
            "stderr": "RuntimeWarning: something minor happened",
            "error": None,
            "warnings": ["RuntimeWarning: something minor happened"],
            "image_base64": None,
        }

        is_valid, feedback = validator.validate_task(task, exec_result, "")

        assert is_valid is True
        assert "validated" in feedback.lower()

    def test_validate_task_result_fails_when_unsolved(self):
        """Final validation should fail if the agent did not mark the task solved."""
        validator = OutputValidator()
        task = {"id": "test_1", "type": "text", "expected_output_contains": ["25", "rows"]}
        exec_result = {
            "status": "success",
            "stdout": "Data has 25 rows",
            "stderr": "",
            "error": None,
            "warnings": [],
            "image_base64": None,
        }

        result = validator.validate_task_result(task, exec_result, "", is_solved=False)

        assert result.passed is False
        assert "did not mark" in result.summary.lower()

    def test_validate_with_visual_feedback_warning(self):
        """Should fail when visual feedback has warning."""
        validator = OutputValidator()

        is_valid, feedback = validator.validate_with_visual_feedback(
            True, "Output validated", "Warning: No image was generated"
        )

        assert is_valid is False

    def test_validate_with_visual_feedback_success(self):
        """Should pass when both validations pass."""
        validator = OutputValidator()

        is_valid, feedback = validator.validate_with_visual_feedback(
            True, "Output validated", "Validated: Chart looks good"
        )

        assert is_valid is True


class TestValidateTaskOutput:
    """Test convenience function."""

    def test_convenience_function(self):
        """Test the convenience function."""
        task = {"id": "test_1", "type": "text", "expected_output_contains": ["test"]}

        exec_result = {"stdout": "This is a test output", "error": None}

        is_valid, feedback = validate_task_output(task, exec_result)

        assert is_valid is True
