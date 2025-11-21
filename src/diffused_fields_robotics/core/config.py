"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Configuration management for diffused fields robotics action primitives.
Handles loading and merging of default primitive parameters with object-specific overrides.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ActionPrimitiveConfigManager:
    """Manages configuration loading for action primitives with proper hierarchy."""
    
    def __init__(self):
        # Get paths relative to this package root
        # Path: src/diffused_fields_robotics/core/config.py -> package root
        self.package_root = Path(__file__).parent.parent.parent.parent
        self.action_primitives_config_path = self.package_root / "config" / "action_primitives.yaml"
        self.pointclouds_config_path = self.package_root / "config" / "pointclouds.yaml"
    
    def load_primitive_defaults(self, primitive_type: str) -> Dict[str, Any]:
        """Load default parameters for a specific primitive type.

        Merges global defaults with primitive-specific configuration.
        Priority: global defaults < primitive-specific config
        """
        if not self.action_primitives_config_path.exists():
            raise FileNotFoundError(f"Action primitives config file not found: {self.action_primitives_config_path}")

        with open(self.action_primitives_config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Get global defaults (if they exist)
        global_defaults = config.get('defaults', {})

        # Get primitive-specific config
        primitive_config = config.get(primitive_type, {})
        if not primitive_config and not global_defaults:
            raise KeyError(f"No default config found for primitive type: {primitive_type}")

        # Merge: global defaults + primitive-specific (primitive-specific takes precedence)
        merged = self.merge_configs(global_defaults, primitive_config)

        return merged
    
    def load_object_overrides(self, object_name: str, primitive_type: str) -> Dict[str, Any]:
        """Load object-specific parameter overrides."""
        if not self.pointclouds_config_path.exists():
            return {}
        
        with open(self.pointclouds_config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Navigate: object_name -> primitive_type -> parameters
        object_config = config.get(object_name, {})
        primitive_overrides = object_config.get(primitive_type, {})
        
        return primitive_overrides
    
    def merge_configs(self, defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge default config with overrides."""
        result = defaults.copy()
        
        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self.merge_configs(result[key], value)
            else:
                # Override or add new parameter
                result[key] = value
        
        return result
    
    def load_merged_config(self, primitive_type: str, object_name: str) -> Dict[str, Any]:
        """Load and merge primitive defaults with object-specific overrides.

        Configuration priority (lowest to highest):
        1. Global defaults (action_primitives.yaml: defaults)
        2. Primitive-specific config (action_primitives.yaml: <primitive_type>)
        3. Object-specific overrides (pointclouds.yaml: <object_name>.<primitive_type>)
        """
        # 1. Load primitive defaults (includes global defaults merged with primitive-specific)
        defaults = self.load_primitive_defaults(primitive_type)

        # 2. Load object-specific overrides
        overrides = self.load_object_overrides(object_name, primitive_type)

        # 3. Merge configurations (overrides take precedence)
        merged_config = self.merge_configs(defaults, overrides)

        return merged_config


def get_action_primitive_config(primitive_type: str, object_name: str) -> Dict[str, Any]:
    """Convenience function to get merged config for a primitive and object."""
    config_manager = ActionPrimitiveConfigManager()
    return config_manager.load_merged_config(primitive_type, object_name)


def get_data_path(relative_path: str) -> Path:
    """Get absolute path to data file relative to package root."""
    config_manager = ActionPrimitiveConfigManager()
    return config_manager.package_root / "data" / relative_path


def get_package_root() -> Path:
    """Get the package root directory."""
    config_manager = ActionPrimitiveConfigManager()
    return config_manager.package_root


def get_plots_dir() -> Path:
    """
    Get the plots directory, creating it if it doesn't exist.

    Returns:
        Path to results/plots directory
    """
    plots_dir = get_package_root() / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def get_results_dir() -> Path:
    """
    Get the results directory, creating it if it doesn't exist.

    Returns:
        Path to results directory
    """
    results_dir = get_package_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_batch_results_path(filename: str) -> Path:
    """
    Get path to batch results file in results/batch_experiments directory.

    Args:
        filename: Name of the batch results pickle file

    Returns:
        Path to batch results file
    """
    batch_results_dir = get_package_root() / "results" / "batch_experiments"
    batch_results_dir.mkdir(parents=True, exist_ok=True)
    return batch_results_dir / filename


def get_ft_data_dir() -> Path:
    """
    Get the force-torque data directory from real-world experiments.

    Returns:
        Path to results/real_world_experiments/ft_data directory
    """
    ft_data_dir = get_package_root() / "results" / "real_world_experiments" / "ft_data"
    ft_data_dir.mkdir(parents=True, exist_ok=True)
    return ft_data_dir


def get_policy_dir() -> Path:
    """
    Get the policy directory containing trained RL models.

    Returns:
        Path to data/policy directory
    """
    policy_dir = get_package_root() / "data" / "policy"
    return policy_dir