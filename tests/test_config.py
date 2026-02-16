"""Tests for configuration parsing."""
import os
import tempfile
import pytest
import yaml

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_parser import load_config


class TestConfigParser:
    """Test suite for config parsing."""

    def test_load_valid_config(self):
        """Should load a valid YAML config file."""
        config_data = {
            'experiment_name': 'test_experiment',
            'seed': 42,
            'agent': {
                'model_id': 'claude-sonnet-4-20250514',
                'temperature': 0.1,
                'max_retries': 3,
                'enable_visual_critic': True
            },
            'sandbox': {
                'template': 'code-interpreter-v1',
                'timeout': 30
            },
            'data': {
                'benchmark_path': 'data/golden_set.json',
                'csv_dir': 'data/csvs/'
            },
            'logging': {
                'log_dir': 'logs/'
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)

            assert config['experiment_name'] == 'test_experiment'
            assert config['seed'] == 42
            assert config['agent']['model_id'] == 'claude-sonnet-4-20250514'
            assert config['agent']['enable_visual_critic'] is True
        finally:
            os.unlink(temp_path)

    def test_load_missing_config(self):
        """Should raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_default_config_file_exists(self):
        """The default config file should exist."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'configs', 'default.yaml')

        assert os.path.exists(config_path), "Default config file should exist"

        config = load_config(config_path)
        assert 'agent' in config
        assert 'sandbox' in config
        assert 'data' in config
