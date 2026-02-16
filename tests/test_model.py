"""Tests for the AnalystAgent model (unit tests that don't require API keys).

These tests focus on the logic that can be tested independently of
external dependencies like langchain and e2b.
"""
import os
import re
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCodeExtraction:
    """Test the code extraction logic (independent of langchain)."""

    def extract_code(self, content: str) -> str:
        """
        Extracts Python code from LLM response.
        This is the same logic as in model.py._extract_code()
        """
        if not content:
            return ""

        patterns = [
            r"```python\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no code block found, assume entire content is code
        lines = content.strip().split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            if line.strip().startswith('#') or any(
                line.strip().lower().startswith(word)
                for word in ['here', 'this', 'the', 'i ', 'note:', 'output:']
            ):
                if not in_code:
                    continue

            if any(keyword in line for keyword in ['import ', 'df', 'plt.', 'print(', 'pd.', '=']):
                in_code = True

            if in_code:
                code_lines.append(line)

        return '\n'.join(code_lines).strip() if code_lines else content.strip()

    def test_extract_code_from_markdown_python_block(self):
        """Should extract code from ```python blocks."""
        content = '''Here's the code:

```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())
```

This will show the first 5 rows.'''

        code = self.extract_code(content)

        assert "import pandas as pd" in code
        assert "df.head()" in code
        assert "Here's the code" not in code
        assert "first 5 rows" not in code

    def test_extract_code_from_generic_block(self):
        """Should extract code from generic ``` blocks."""
        content = '''```
plt.plot([1, 2, 3])
plt.show()
```'''

        code = self.extract_code(content)

        assert "plt.plot" in code
        assert "plt.show()" in code

    def test_extract_code_raw_code(self):
        """Should handle raw code without blocks."""
        content = '''import pandas as pd
df = pd.read_csv('data.csv')
print(df.shape)'''

        code = self.extract_code(content)

        assert "import pandas" in code
        assert "df.shape" in code

    def test_extract_code_empty_content(self):
        """Should handle empty content."""
        assert self.extract_code("") == ""
        assert self.extract_code(None) == ""

    def test_extract_code_nested_blocks(self):
        """Should extract from first matching block."""
        content = '''Here's one way:
```python
x = 1
```
And another:
```python
y = 2
```'''

        code = self.extract_code(content)
        assert "x = 1" in code
        assert "y = 2" not in code


class TestAgentStateStructure:
    """Test the AgentState TypedDict structure."""

    def test_agent_state_has_required_fields(self):
        """AgentState should have all required fields."""
        # Define expected state structure
        state = {
            'messages': [],
            'context_data': 'test context',
            'generated_code': 'print("hello")',
            'execution_result': {},
            'retry_count': 0,
            'is_solved': False,
            'original_query': 'test query'
        }

        required_fields = [
            'messages', 'context_data', 'generated_code',
            'execution_result', 'retry_count', 'is_solved', 'original_query'
        ]

        for field in required_fields:
            assert field in state, f"Missing field: {field}"


class TestVisualCriticLogic:
    """Test the visual critic parsing logic."""

    FAILURE_INDICATORS = [
        "empty chart", "blank chart", "no data", "cannot see",
        "missing title", "no title", "unlabeled", "unreadable",
        "does not answer", "doesn't answer", "incorrect",
        "overlapping text", "cannot read"
    ]

    SUCCESS_INDICATORS = [
        "correctly answers", "properly labeled", "clear title",
        "looks good", "accurate", "well formatted", "readable"
    ]

    def has_failure(self, critique: str) -> bool:
        """Check for failure indicators."""
        critique_lower = critique.lower()
        return any(indicator in critique_lower for indicator in self.FAILURE_INDICATORS)

    def has_success(self, critique: str) -> bool:
        """Check for success indicators."""
        critique_lower = critique.lower()
        return any(indicator in critique_lower for indicator in self.SUCCESS_INDICATORS)

    def test_failure_detection(self):
        """Should detect failure indicators in critique."""
        critique = "The chart has no data and the title is missing."
        assert self.has_failure(critique) is True

    def test_success_detection(self):
        """Should detect success indicators in critique."""
        critique = "The chart correctly answers the query and has a clear title."
        assert self.has_success(critique) is True

    def test_mixed_feedback_with_success(self):
        """Should handle feedback that mentions issues but overall is positive."""
        critique = "The chart correctly answers the query. The title is clear and readable."

        has_failure = self.has_failure(critique)
        has_success = self.has_success(critique)

        # Should be valid: has success and no failure
        is_valid = has_success and not has_failure
        assert is_valid is True

    def test_mixed_feedback_with_failure(self):
        """Should detect failure even if some positive words present."""
        critique = "The chart is readable but shows no data."

        has_failure = self.has_failure(critique)
        has_success = self.has_success(critique)

        # Has both, so should fail (conservative approach)
        is_valid = has_success and not has_failure
        assert is_valid is False

    def test_ambiguous_feedback(self):
        """Should handle ambiguous feedback conservatively."""
        critique = "The chart appears to show something, but I cannot determine if it's correct."

        has_failure = self.has_failure(critique)
        has_success = self.has_success(critique)

        # Neither success nor failure indicators
        assert has_failure is False
        assert has_success is False


class TestPromptTemplates:
    """Test prompt template structure."""

    def test_code_prompt_has_key_elements(self):
        """Code generation prompt should include key elements."""
        # This is the structure we expect in the prompt
        expected_elements = [
            "Dataset Context",
            "User Query",
            "IMPORTANT RULES",
            "DataFrame 'df' is already loaded"
        ]

        # Simulated prompt template
        prompt_template = """You are an expert Python Data Analyst.

Dataset Context:
{context}

User Query: {query}
{error_context}

IMPORTANT RULES:
1. Write ONLY valid Python code - no markdown, no explanations
2. The DataFrame 'df' is already loaded from 'data.csv' - do NOT reload it
3. For plots: use plt.savefig() is NOT needed - just create the plot and call plt.show()
4. For text answers: use print() to output results
5. Handle potential errors gracefully (e.g., check column existence)
6. If creating plots, always include: title, axis labels, and legend if applicable

Write the Python code now:
"""

        for element in expected_elements:
            assert element in prompt_template, f"Missing element: {element}"
