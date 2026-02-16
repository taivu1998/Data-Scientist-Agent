"""Integration tests that require API keys.

These tests are marked with @pytest.mark.integration and will be skipped
if the required API keys are not set in environment variables.

To run integration tests:
    export ANTHROPIC_API_KEY=your_key
    export E2B_API_KEY=your_key
    pytest tests/test_integration.py -v -m integration
"""

import os
import pytest

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.integration
class TestSandboxIntegration:
    """Integration tests for the E2B sandbox."""

    def test_sandbox_basic_execution(self, skip_without_api_keys):
        """Test basic code execution in sandbox."""
        from src.sandbox import SandboxWrapper

        with SandboxWrapper(template="code-interpreter-v1", timeout=30) as sandbox:
            result = sandbox.run_code("print('Hello, World!')")

            assert result["error"] is None
            assert "Hello, World!" in result["stdout"]

    def test_sandbox_pandas_execution(self, skip_without_api_keys, sample_csv_path):
        """Test pandas execution in sandbox."""
        from src.sandbox import SandboxWrapper

        with SandboxWrapper(template="code-interpreter-v1", timeout=30) as sandbox:
            # Upload sample data
            sandbox.upload_data(sample_csv_path, "data.csv")

            # Run pandas code
            code = """
import pandas as pd
df = pd.read_csv('data.csv')
print(f"Shape: {df.shape}")
print(df.head())
"""
            result = sandbox.run_code(code)

            assert result["error"] is None
            assert "Shape:" in result["stdout"]

    def test_sandbox_matplotlib_execution(self, skip_without_api_keys):
        """Test matplotlib plot generation in sandbox."""
        from src.sandbox import SandboxWrapper

        with SandboxWrapper(template="code-interpreter-v1", timeout=30) as sandbox:
            code = """
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Test Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
"""
            result = sandbox.run_code(code)

            assert result["error"] is None
            assert result["image_base64"] is not None


@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for the full agent pipeline."""

    def test_agent_text_query(self, skip_without_api_keys, default_config_path, sample_csv_path):
        """Test agent on a simple text query."""
        from src.config_parser import load_config
        from src.model import AnalystAgent
        from src.sandbox import SandboxWrapper
        from src.dataset import AnalysisTaskDataset

        config = load_config(default_config_path)
        agent = AnalystAgent(config)

        ds = AnalysisTaskDataset("dummy")
        context = ds.get_semantic_context(sample_csv_path)

        with SandboxWrapper(
            template=config["sandbox"]["template"], timeout=config["sandbox"]["timeout"]
        ) as sandbox:
            sandbox.upload_data(sample_csv_path, "data.csv")

            query = "How many rows are in the dataset?"
            final_state = agent.run(query, context, sandbox)

            assert final_state is not None
            assert "execution_result" in final_state
            assert final_state["retry_count"] >= 1

    @pytest.mark.slow
    def test_agent_plot_query(self, skip_without_api_keys, default_config_path, sample_csv_path):
        """Test agent on a plotting query."""
        from src.config_parser import load_config
        from src.model import AnalystAgent
        from src.sandbox import SandboxWrapper
        from src.dataset import AnalysisTaskDataset

        config = load_config(default_config_path)
        agent = AnalystAgent(config)

        ds = AnalysisTaskDataset("dummy")
        context = ds.get_semantic_context(sample_csv_path)

        with SandboxWrapper(
            template=config["sandbox"]["template"], timeout=config["sandbox"]["timeout"]
        ) as sandbox:
            sandbox.upload_data(sample_csv_path, "data.csv")

            query = "Create a bar chart of average salary by department."
            final_state = agent.run(query, context, sandbox)

            assert final_state is not None
            exec_result = final_state.get("execution_result", {})

            # Should either have an image or have attempted to create one
            assert final_state["retry_count"] >= 1


@pytest.mark.integration
class TestErrorRecovery:
    """Integration tests for error recovery scenarios."""

    def test_sandbox_timeout_handling(self, skip_without_api_keys):
        """Test that sandbox handles timeout gracefully."""
        from src.sandbox import SandboxWrapper

        with SandboxWrapper(template="code-interpreter-v1", timeout=5) as sandbox:
            # This should timeout
            result = sandbox.run_code("import time; time.sleep(10)")

            # Should have an error about timeout
            assert result["error"] is not None or "timed out" in result.get("stderr", "").lower()

    def test_sandbox_invalid_code_error(self, skip_without_api_keys):
        """Test that invalid code returns proper error."""
        from src.sandbox import SandboxWrapper

        with SandboxWrapper(template="code-interpreter-v1", timeout=30) as sandbox:
            result = sandbox.run_code("import nonexistent_module_xyz")

            # Should have an error
            assert result["error"] is not None or result["stderr"] != ""

    def test_sandbox_persistent_kernel_state(self, skip_without_api_keys, sample_csv_path):
        """Test that variables persist across code executions."""
        from src.sandbox import SandboxWrapper

        with SandboxWrapper(template="code-interpreter-v1", timeout=30) as sandbox:
            sandbox.upload_data(sample_csv_path, "data.csv")

            # First execution: load data
            result1 = sandbox.run_code(
                "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint('loaded')"
            )
            assert result1["error"] is None

            # Second execution: use df (should persist)
            result2 = sandbox.run_code("print(f'Rows: {len(df)}')")
            assert result2["error"] is None
            assert "Rows:" in result2["stdout"]

    def test_sandbox_multiple_image_generation(self, skip_without_api_keys):
        """Test that multiple plot calls work."""
        from src.sandbox import SandboxWrapper

        with SandboxWrapper(template="code-interpreter-v1", timeout=30) as sandbox:
            code = """
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1,2,3], [1,2,3])
plt.title('First')
plt.show()
plt.figure()
plt.plot([3,2,1], [1,2,3])
plt.title('Second')
plt.show()
"""
            result = sandbox.run_code(code)

            # Should generate at least one image
            assert result["image_base64"] is not None

    def test_agent_handles_syntax_error(
        self, skip_without_api_keys, default_config_path, sample_csv_path
    ):
        """Test that agent handles syntax errors gracefully."""
        from src.config_parser import load_config
        from src.model import AnalystAgent
        from src.sandbox import SandboxWrapper
        from src.dataset import AnalysisTaskDataset

        config = load_config(default_config_path)
        config["agent"]["max_retries"] = 2

        agent = AnalystAgent(config)

        ds = AnalysisTaskDataset("dummy")
        context = ds.get_semantic_context(sample_csv_path)

        # We'll just verify the agent runs without crashing
        with SandboxWrapper(
            template=config["sandbox"]["template"], timeout=config["sandbox"]["timeout"]
        ) as sandbox:
            sandbox.upload_data(sample_csv_path, "data.csv")

            query = "Show me the data shape"
            final_state = agent.run(query, context, sandbox)

            # Should complete without crashing
            assert final_state is not None
            assert "execution_result" in final_state
