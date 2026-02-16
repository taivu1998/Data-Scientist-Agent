"""Pytest configuration and shared fixtures."""

import os
import pytest


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require API keys)"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow running")


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def sample_csv_path(project_root):
    """Return path to the sample salaries.csv file."""
    return os.path.join(project_root, "data", "csvs", "salaries.csv")


@pytest.fixture
def default_config_path(project_root):
    """Return path to the default config file."""
    return os.path.join(project_root, "configs", "default.yaml")


@pytest.fixture
def has_anthropic_key():
    """Check if ANTHROPIC_API_KEY is set."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


@pytest.fixture
def has_e2b_key():
    """Check if E2B_API_KEY is set."""
    return bool(os.getenv("E2B_API_KEY"))


@pytest.fixture
def skip_without_api_keys():
    """Skip test if API keys are not available."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    e2b_key = os.getenv("E2B_API_KEY")

    if not anthropic_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    if not e2b_key:
        pytest.skip("E2B_API_KEY not set")

    return True
