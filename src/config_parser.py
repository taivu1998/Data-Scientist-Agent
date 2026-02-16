import argparse
import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args() -> Dict[str, Any]:
    """
    Parses command line arguments and overrides YAML config values.
    
    Returns:
        Dict[str, Any]: Merged configuration dictionary.
    """
    parser = argparse.ArgumentParser(description="Auto-Analyst Research Runner")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    parser.add_argument('--query', type=str, help="Override query for inference")
    parser.add_argument('--csv_path', type=str, help="Override CSV path for inference")
    
    # Allow overriding specific config keys via CLI
    parser.add_argument('--model_id', type=str, help="Override model ID")
    parser.add_argument('--enable_visual_critic', type=str, help="Toggle visual critic (true/false)")

    args = parser.parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.model_id:
        config['agent']['model_id'] = args.model_id
    if args.enable_visual_critic:
        config['agent']['enable_visual_critic'] = (args.enable_visual_critic.lower() == 'true')
    
    # Add runtime args
    config['runtime'] = {}
    if args.query:
        config['runtime']['query'] = args.query
    if args.csv_path:
        config['runtime']['csv_path'] = args.csv_path

    return config