import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    stage: str
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        parts = []
        if self.reasons:
            parts.append("; ".join(self.reasons))
        if self.warnings:
            parts.append(f"Warnings: {'; '.join(self.warnings)}")
        if not parts:
            parts.append("Validation passed")
        return " | ".join(parts)

    def as_tuple(self) -> Tuple[bool, str]:
        return self.passed, self.summary


class OutputValidator:
    """
    Validate agent output against golden-set expectations.
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

    def validate_task(
        self, task: Dict[str, Any], exec_result: Dict[str, Any], generated_code: str = ""
    ) -> Tuple[bool, str]:
        return self.validate_task_detailed(task, exec_result, generated_code).as_tuple()

    def validate_task_detailed(
        self, task: Dict[str, Any], exec_result: Dict[str, Any], generated_code: str = ""
    ) -> ValidationResult:
        task_type = task.get("type", "text")

        if self._is_execution_failure(exec_result):
            return ValidationResult(
                passed=False,
                stage="execution",
                reasons=[
                    f"Execution error: {exec_result.get('error') or exec_result.get('stderr')}"
                ],
                warnings=self._extract_warnings(exec_result),
                evidence={"status": exec_result.get("status")},
            )

        if task_type in {"plot", "plot_log_check"}:
            return self._validate_plot(task, exec_result, generated_code)
        return self._validate_text(task, exec_result)

    def validate_task_result(
        self,
        task: Dict[str, Any],
        exec_result: Dict[str, Any],
        generated_code: str,
        is_solved: bool,
    ) -> ValidationResult:
        base = self.validate_task_detailed(task, exec_result, generated_code)
        if not base.passed:
            return base

        if not is_solved:
            return ValidationResult(
                passed=False,
                stage="solve_state",
                reasons=["Agent did not mark the task as solved"],
                warnings=base.warnings,
                evidence={
                    **base.evidence,
                    "critic_status": exec_result.get("critic_status"),
                    "visual_feedback": exec_result.get("visual_feedback"),
                },
            )

        return ValidationResult(
            passed=True,
            stage="final",
            reasons=base.reasons or ["Validation passed"],
            warnings=base.warnings,
            evidence={
                **base.evidence,
                "critic_status": exec_result.get("critic_status"),
                "visual_feedback": exec_result.get("visual_feedback"),
            },
        )

    def _validate_text(self, task: Dict[str, Any], exec_result: Dict[str, Any]) -> ValidationResult:
        expected = task.get("expected_output_contains", [])
        warnings = self._extract_warnings(exec_result)

        if not expected:
            return ValidationResult(
                passed=True,
                stage="text",
                reasons=["No specific output expected, assuming success"],
                warnings=warnings,
            )

        stdout = exec_result.get("stdout", "")
        missing = [expected_str for expected_str in expected if expected_str.lower() not in stdout.lower()]
        if missing:
            return ValidationResult(
                passed=False,
                stage="text",
                reasons=[
                    f"Expected output containing '{missing[0]}' not found. Got: {stdout[:200]}"
                ],
                warnings=warnings,
                evidence={"missing": missing},
            )

        return ValidationResult(
            passed=True,
            stage="text",
            reasons=["Text output validated: all expected strings found"],
            warnings=warnings,
            evidence={"expected_output_contains": expected},
        )

    def _validate_plot(
        self, task: Dict[str, Any], exec_result: Dict[str, Any], generated_code: str
    ) -> ValidationResult:
        warnings = self._extract_warnings(exec_result)
        if not exec_result.get("image_base64"):
            return ValidationResult(
                passed=False,
                stage="plot",
                reasons=["No image was generated"],
                warnings=warnings,
            )

        expected_visual = task.get("expected_visual", "")
        if not expected_visual:
            return ValidationResult(
                passed=True,
                stage="plot",
                reasons=["No specific plot type expected"],
                warnings=warnings,
            )

        if expected_visual == "scatter_log":
            return self._validate_log_scale(generated_code, warnings)

        return self._validate_plot_type(expected_visual, generated_code, warnings)

    def _validate_plot_type(
        self, expected_visual: str, generated_code: str, warnings: List[str]
    ) -> ValidationResult:
        code_lower = generated_code.lower()
        expected_keywords = self.PLOT_TYPE_KEYWORDS.get(expected_visual, [expected_visual])
        found = [keyword for keyword in expected_keywords if keyword in code_lower]

        if found:
            return ValidationResult(
                passed=True,
                stage="plot_type",
                reasons=[f"Plot type validated: {expected_visual}"],
                warnings=warnings,
                evidence={"matched_keywords": found},
            )

        return ValidationResult(
            passed=False,
            stage="plot_type",
            reasons=[
                f"Expected plot type '{expected_visual}' not found in code. "
                f"Found keywords: {found}"
            ],
            warnings=warnings,
            evidence={"expected_keywords": expected_keywords},
        )

    def _validate_log_scale(self, generated_code: str, warnings: List[str]) -> ValidationResult:
        code_lower = generated_code.lower()
        has_log_scale = any(re.search(pattern, code_lower) for pattern in self.LOG_SCALE_PATTERNS)

        if not has_log_scale:
            return ValidationResult(
                passed=False,
                stage="plot_log_check",
                reasons=["Log scale not applied (expected_visual: scatter_log)"],
                warnings=warnings,
            )

        return ValidationResult(
            passed=True,
            stage="plot_log_check",
            reasons=["Log scale validated"],
            warnings=warnings,
        )

    def validate_with_visual_feedback(
        self, is_valid: bool, validation_feedback: str, visual_feedback: Optional[str]
    ) -> Tuple[bool, str]:
        if not is_valid:
            return False, f"Output validation failed: {validation_feedback}"

        if visual_feedback and "Warning" in visual_feedback:
            return False, f"Visual feedback warning: {visual_feedback}"

        if is_valid and (not visual_feedback or "Validated" in visual_feedback):
            return True, f"Validation passed: {validation_feedback}"

        return is_valid, validation_feedback

    def _is_execution_failure(self, exec_result: Dict[str, Any]) -> bool:
        status = exec_result.get("status")
        return status == "error" or bool(exec_result.get("error"))

    def _extract_warnings(self, exec_result: Dict[str, Any]) -> List[str]:
        warnings = list(exec_result.get("warnings", []))
        if exec_result.get("status") == "warning" and exec_result.get("stderr"):
            stderr = exec_result["stderr"]
            if stderr not in warnings:
                warnings.append(stderr)
        return warnings


def validate_task_output(
    task: Dict[str, Any],
    exec_result: Dict[str, Any],
    generated_code: str = "",
    visual_feedback: Optional[str] = None,
) -> Tuple[bool, str]:
    validator = OutputValidator()
    is_valid, validation_feedback = validator.validate_task(task, exec_result, generated_code)

    if not is_valid:
        return False, validation_feedback

    if visual_feedback and "Warning" in visual_feedback:
        return False, f"Visual warning: {visual_feedback}"

    return is_valid, validation_feedback
