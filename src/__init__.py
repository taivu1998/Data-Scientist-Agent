"""
Auto-Analyst: A Research-Grade Data Analysis Agent

This package implements a graph-based autonomous agent for data analysis
with multimodal visual verification.

Architecture: Plan -> Code -> Execute -> VisualVerify -> Refine

Key Components:
- model.py: AnalystAgent - the core LangGraph-based agent
- sandbox.py: SandboxWrapper - E2B Firecracker microVM integration
- dataset.py: AnalysisTaskDataset - Golden Set benchmark loader
- trainer.py: Trainer - rigorous evaluation framework
"""

__version__ = "0.1.0"

# Lazy imports to avoid requiring all dependencies at import time
# This allows tests to run without langchain/e2b installed

def __getattr__(name):
    """Lazy import of heavy dependencies."""
    if name == "AnalystAgent":
        from src.model import AnalystAgent
        return AnalystAgent
    elif name == "AgentState":
        from src.model import AgentState
        return AgentState
    elif name == "SandboxWrapper":
        from src.sandbox import SandboxWrapper
        return SandboxWrapper
    elif name == "AnalysisTaskDataset":
        from src.dataset import AnalysisTaskDataset
        return AnalysisTaskDataset
    elif name == "Trainer":
        from src.trainer import Trainer
        return Trainer
    elif name == "parse_args":
        from src.config_parser import parse_args
        return parse_args
    elif name == "load_config":
        from src.config_parser import load_config
        return load_config
    elif name == "seed_everything":
        from src.utils import seed_everything
        return seed_everything
    elif name == "setup_logger":
        from src.utils import setup_logger
        return setup_logger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AnalystAgent",
    "AgentState",
    "SandboxWrapper",
    "AnalysisTaskDataset",
    "Trainer",
    "parse_args",
    "load_config",
    "seed_everything",
    "setup_logger",
]
