import logging
import re
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)


class OutputValidator:
    """
    Validates agent output against golden set expectations.

    Handles:
    - Text output validation (expected_output_contains)
    - Plot type validation (expected_visual)
    - Combined validation with execution results
    """

    PLOT_TYPE_KEYWORDS = {
        "bar_chart": ["bar", "barplot", "bar chart"],
        "histogram": ["hist", "histogram"],
        "scatter": ["scatter", "scatter plot"],
        "boxplot": ["box", "boxplot", "box plot"],
        "heatmap": ["heatmap", "corr", "correlation"],
        "line": ["line", "plot", "line plot"],
        "pie": ["pie", "pie chart"],
    }

    LOG_SCALE_PATTERNS = [
        r"log\.scale|logscale|set_yscale\(['\"]log",
        r"yscale\(['\"]log",
        r"plt\.yscale\(['\"]log",
    ]

    def __init__(self):
        pass

    def validate_task(
        self, task: Dict[str, Any], exec_result: Dict[str, Any], generated_code: str = ""
    ) -> Tuple[bool, str]:
        """
        Validate agent output against task expectations.

        Args:
            task: Task dict with expected_output_contains, expected_visual, type
            exec_result: Execution result from sandbox
            generated_code: The code that was executed

        Returns:
            Tuple of (is_valid, feedback_message)
        """
        task_type = task.get("type", "text")

        if exec_result.get("error") or exec_result.get("stderr"):
            return (
                False,
                f"Execution error: {exec_result.get('error') or exec_result.get('stderr')}",
            )

        if task_type in ["plot", "plot_log_check"]:
            return self._validate_plot(task, exec_result, generated_code)
        else:
            return self._validate_text(task, exec_result)

    def _validate_text(self, task: Dict[str, Any], exec_result: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate text output against expected_output_contains.
        """
        expected = task.get("expected_output_contains", [])

        if not expected:
            return True, "No specific output expected, assuming success"

        stdout = exec_result.get("stdout", "")

        for expected_str in expected:
            if expected_str.lower() not in stdout.lower():
                return (
                    False,
                    f"Expected output containing '{expected_str}' not found. Got: {stdout[:200]}",
                )

        return True, f"Text output validated: all expected strings found"

    def _validate_plot(
        self, task: Dict[str, Any], exec_result: Dict[str, Any], generated_code: str
    ) -> Tuple[bool, str]:
        """
        Validate plot output:
        1. Check if an image was generated
        2. Check if plot type matches expected_visual
        3. Check for log scale if required
        """
        if not exec_result.get("image_base64"):
            return False, "No image was generated"

        expected_visual = task.get("expected_visual", "")

        if not expected_visual:
            return True, "No specific plot type expected"

        if expected_visual == "scatter_log":
            return self._validate_log_scale(task, generated_code)

        return self._validate_plot_type(expected_visual, generated_code)

    def _validate_plot_type(self, expected_visual: str, generated_code: str) -> Tuple[bool, str]:
        """
        Validate that the generated code creates the expected plot type.
        """
        code_lower = generated_code.lower()

        expected_keywords = self.PLOT_TYPE_KEYWORDS.get(expected_visual, [expected_visual])

        found = any(keyword in code_lower for keyword in expected_keywords)

        if found:
            return True, f"Plot type validated: {expected_visual}"

        return (
            False,
            f"Expected plot type '{expected_visual}' not found in code. Found keywords: {[k for k in expected_keywords if k in code_lower]}",
        )

    def _validate_log_scale(self, task: Dict[str, Any], generated_code: str) -> Tuple[bool, str]:
        """
        Validate log scale is applied for plot_log_check tasks.
        """
        code_lower = generated_code.lower()

        has_log_scale = any(re.search(pattern, code_lower) for pattern in self.LOG_SCALE_PATTERNS)

        if not has_log_scale:
            return False, "Log scale not applied (expected_visual: scatter_log)"

        return True, "Log scale validated"

    def validate_with_visual_feedback(
        self, is_valid: bool, validation_feedback: str, visual_feedback: Optional[str]
    ) -> Tuple[bool, str]:
        """
        Combine output validation with visual critic feedback.

        Both must pass for overall success.
        """
        if not is_valid:
            return False, f"Output validation failed: {validation_feedback}"

        if visual_feedback and "Warning" in visual_feedback:
            return False, f"Visual feedback warning: {visual_feedback}"

        if is_valid and (not visual_feedback or "Validated" in visual_feedback):
            return True, f"Validation passed: {validation_feedback}"

        return is_valid, validation_feedback


def validate_task_output(
    task: Dict[str, Any],
    exec_result: Dict[str, Any],
    generated_code: str = "",
    visual_feedback: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Convenience function to validate task output.

    Args:
        task: Task dictionary with expected outputs
        exec_result: Execution result from sandbox
        generated_code: The code that was executed
        visual_feedback: Feedback from visual critic (if any)

    Returns:
        Tuple of (is_valid, feedback_message)
    """
    validator = OutputValidator()

    is_valid, validation_feedback = validator.validate_task(task, exec_result, generated_code)

    if not is_valid:
        return False, validation_feedback

    if visual_feedback and "Warning" in visual_feedback:
        return False, f"Visual warning: {visual_feedback}"

    return is_valid, validation_feedback
