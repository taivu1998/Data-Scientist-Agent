import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SandboxWrapper:
    """
    Wrapper around E2B Code Interpreter to handle file uploads
    and secure code execution in Firecracker microVMs.
    """

    def __init__(self, template: str = "code-interpreter-v1", timeout: int = 30):
        self.template = template
        self.timeout = timeout
        self.sandbox: Optional[Any] = None

    def __enter__(self):
        """Start the sandbox session."""
        logger.info(f"Starting E2B sandbox with template: {self.template}")
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError as exc:  # pragma: no cover - exercised in dependency-light environments
            raise RuntimeError(
                "SandboxWrapper requires 'e2b-code-interpreter' to start a sandbox session."
            ) from exc

        self.sandbox = Sandbox(template=self.template)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up sandbox resources."""
        if self.sandbox:
            try:
                self.sandbox.close()
                logger.info("Sandbox closed successfully")
            except Exception as e:  # pragma: no cover - defensive cleanup
                logger.warning(f"Error closing sandbox: {e}")

    def upload_data(self, local_path: str, remote_path: str = "data.csv") -> str:
        """Upload a local file to the sandbox filesystem."""
        if not self.sandbox:
            raise RuntimeError("Sandbox is not active. Use within 'with' context.")

        try:
            with open(local_path, "rb") as f:
                file_content = f.read()

            self.sandbox.files.write(remote_path, file_content)
            logger.info(f"Uploaded {local_path} -> {remote_path}")
            return remote_path

        except FileNotFoundError:
            logger.error(f"Local file not found: {local_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            raise

    def run_code(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in the persistent Jupyter kernel and normalize the result.
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox is not active. Use within 'with' context.")

        result = {
            "status": "success",
            "stdout": "",
            "stderr": "",
            "error": None,
            "image_base64": None,
            "warnings": [],
        }

        try:
            execution = self.sandbox.run_code(code, timeout=self.timeout)

            if execution.logs:
                result["stdout"] = "".join(execution.logs.stdout) if execution.logs.stdout else ""
                result["stderr"] = "".join(execution.logs.stderr) if execution.logs.stderr else ""

            if execution.error:
                result["error"] = str(execution.error)
                result["status"] = "error"
                logger.warning(f"Code execution error: {execution.error}")
            elif result["stderr"]:
                result["status"] = self._classify_stderr(result["stderr"])
                if result["status"] == "warning":
                    result["warnings"] = [result["stderr"]]
                else:
                    result["error"] = result["stderr"]

            if execution.results:
                for artifact in execution.results:
                    if hasattr(artifact, "png") and artifact.png:
                        result["image_base64"] = artifact.png
                        logger.info("Plot captured from execution")
                        break

        except TimeoutError:
            result["status"] = "error"
            result["error"] = f"Execution timed out after {self.timeout} seconds"
            result["stderr"] = result["error"]
            logger.error(result["error"])
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["stderr"] = str(e)
            logger.error(f"Unexpected execution error: {e}")

        return result

    @staticmethod
    def _classify_stderr(stderr: str) -> str:
        """Classify stderr as a warning or an execution error."""
        stderr_lower = stderr.lower()

        warning_markers = (
            "warning:",
            "warning ",
            "runtimewarning",
            "userwarning",
            "futurewarning",
            "deprecationwarning",
            "syntaxwarning",
            "resourcewarning",
        )
        error_markers = (
            "traceback",
            "error:",
            "exception",
            "module not found",
            "modulenotfounderror",
            "importerror",
            "nameerror",
            "typeerror",
            "valueerror",
            "attributeerror",
            "indexerror",
            "keyerror",
            "syntaxerror",
            "zerodivisionerror",
            "filenotfounderror",
            "permissionerror",
            "assertionerror",
        )

        has_warning_marker = any(marker in stderr_lower for marker in warning_markers)
        has_error_marker = any(marker in stderr_lower for marker in error_markers)

        if has_error_marker:
            return "error"
        if has_warning_marker:
            return "warning"
        return "error"

    def list_files(self, path: str = "/") -> List[str]:
        """List files in the sandbox filesystem."""
        if not self.sandbox:
            raise RuntimeError("Sandbox is not active.")

        try:
            files = self.sandbox.files.list(path)
            return [f.name for f in files]
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []

    def read_file(self, remote_path: str) -> Optional[bytes]:
        """Read a file from the sandbox filesystem."""
        if not self.sandbox:
            raise RuntimeError("Sandbox is not active.")

        try:
            content = self.sandbox.files.read(remote_path)
            return content
        except Exception as e:
            logger.error(f"Failed to read file {remote_path}: {e}")
            return None
