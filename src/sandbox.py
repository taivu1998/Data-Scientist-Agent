import logging
from typing import Dict, Any, Optional, List

from e2b_code_interpreter import Sandbox

logger = logging.getLogger(__name__)


class SandboxWrapper:
    """
    Wrapper around E2B Code Interpreter to handle file uploads
    and secure code execution in Firecracker microVMs.

    The E2B sandbox provides:
    - Isolated execution in Firecracker microVMs (AWS Lambda-grade isolation)
    - Persistent Jupyter kernel session (variables survive across code blocks)
    - Automatic cleanup on context exit
    """

    def __init__(self, template: str = "code-interpreter-v1", timeout: int = 30):
        """
        Args:
            template: The E2B sandbox template (default: code-interpreter-v1)
            timeout: Execution timeout in seconds per code block
        """
        self.template = template
        self.timeout = timeout
        self.sandbox: Optional[Sandbox] = None

    def __enter__(self):
        """Start the sandbox session."""
        logger.info(f"Starting E2B sandbox with template: {self.template}")
        self.sandbox = Sandbox(template=self.template)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up sandbox resources."""
        if self.sandbox:
            try:
                self.sandbox.close()
                logger.info("Sandbox closed successfully")
            except Exception as e:
                logger.warning(f"Error closing sandbox: {e}")

    def upload_data(self, local_path: str, remote_path: str = "data.csv") -> str:
        """
        Uploads a local file to the sandbox filesystem.

        Args:
            local_path: Path to the local file to upload
            remote_path: Destination path in the sandbox (default: data.csv)

        Returns:
            The remote path where the file was uploaded

        Raises:
            RuntimeError: If sandbox is not active
            FileNotFoundError: If local file doesn't exist
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox is not active. Use within 'with' context.")

        try:
            # Read file content as bytes and upload
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
        Executes Python code in the persistent Jupyter kernel.

        The kernel maintains state between calls, so variables defined
        in one execution are available in subsequent calls.

        Args:
            code: Python code to execute

        Returns:
            Dict containing:
                - stdout: Standard output from execution
                - stderr: Standard error output
                - error: Any execution error message
                - image_base64: Base64-encoded PNG if a plot was generated
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox is not active. Use within 'with' context.")

        result = {
            "stdout": "",
            "stderr": "",
            "error": None,
            "image_base64": None
        }

        try:
            execution = self.sandbox.run_code(code, timeout=self.timeout)

            # Extract logs
            if execution.logs:
                result["stdout"] = "".join(execution.logs.stdout) if execution.logs.stdout else ""
                result["stderr"] = "".join(execution.logs.stderr) if execution.logs.stderr else ""

            # Extract error if present
            if execution.error:
                result["error"] = str(execution.error)
                logger.warning(f"Code execution error: {execution.error}")

            # Extract matplotlib plots if generated
            if execution.results:
                for artifact in execution.results:
                    # E2B returns PNG data in the png attribute
                    if hasattr(artifact, "png") and artifact.png:
                        result["image_base64"] = artifact.png
                        logger.info("Plot captured from execution")
                        break

        except TimeoutError:
            result["error"] = f"Execution timed out after {self.timeout} seconds"
            result["stderr"] = result["error"]
            logger.error(result["error"])
        except Exception as e:
            result["error"] = str(e)
            result["stderr"] = str(e)
            logger.error(f"Unexpected execution error: {e}")

        return result

    def list_files(self, path: str = "/") -> List[str]:
        """
        Lists files in the sandbox filesystem.

        Args:
            path: Directory path to list (default: root)

        Returns:
            List of file/directory names
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox is not active.")

        try:
            files = self.sandbox.files.list(path)
            return [f.name for f in files]
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []

    def read_file(self, remote_path: str) -> Optional[bytes]:
        """
        Reads a file from the sandbox filesystem.

        Args:
            remote_path: Path to file in sandbox

        Returns:
            File contents as bytes, or None on error
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox is not active.")

        try:
            content = self.sandbox.files.read(remote_path)
            return content
        except Exception as e:
            logger.error(f"Failed to read file {remote_path}: {e}")
            return None
