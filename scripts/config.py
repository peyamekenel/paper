"""Configuration management for the recommendation system."""
import os
from typing import Any, Dict, Optional
import yaml


class Config:
    """Configuration manager that loads from YAML and allows CLI overrides."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from YAML file."""
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Example:
            config.get('model.name') -> 'all-MiniLM-L6-v2'
            config.get('similarity.alpha') -> 0.5
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Example:
            config.set('model.name', 'all-mpnet-base-v2')
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update_from_args(self, args: Any) -> None:
        """Update config from command-line arguments."""
        # Map CLI args to config keys
        arg_mapping = {
            'model': 'model.name',
            'batch_size': 'model.batch_size',
            'max_length': 'model.max_length',
            'k': 'recommendations.default_k',
            'alpha': 'similarity.alpha',
            'outdir': 'paths.output_dir',
        }
        
        for arg_name, config_key in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                self.set(config_key, getattr(args, arg_name))
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access: config['model.name']"""
        return self.get(key)
    
    def __repr__(self) -> str:
        """String representation of config."""
        return f"Config({self.config_path})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return self._config.copy()


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from file."""
    return Config(config_path)

