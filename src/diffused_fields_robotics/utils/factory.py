"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Factory functions for creating action primitive controllers.

This module provides factory patterns to standardize controller creation
and eliminate repetitive initialization code.
"""

import numpy as np
from typing import Optional, Any, Dict, Union

# Import the core classes - action primitives have been removed
try:
    from diffused_fields import Pointcloud
except ImportError:
    print("Warning: diffused_fields not available. Some functionality may be limited.")
    Pointcloud = None


def create_primitive_controller(
    primitive_type: str,
    pcloud: Union[Pointcloud, str],
    diffusion_scalar: Optional[float] = None,
    source_vertices: Optional[np.ndarray] = None,
    start_vertex: Optional[int] = None,
    end_vertex: Optional[int] = None,
    **kwargs
) -> Any:
    """
    Factory function to create action primitive controllers.
    
    Args:
        primitive_type: Type of primitive ('cutting', 'slicing', etc.)
        pcloud: Point cloud object or filename
        diffusion_scalar: Optional diffusion scalar override
        source_vertices: Optional source vertices array
        start_vertex: Optional start vertex index
        end_vertex: Optional end vertex index
        **kwargs: Additional arguments passed to controller constructor
        
    Returns:
        Initialized action primitive controller
        
    Raises:
        ValueError: If primitive_type is not supported
        TypeError: If pcloud is neither Pointcloud nor string
        ImportError: If action primitives cannot be imported
    """
    # Import action primitives dynamically
    from ..utils.experiment_base import _import_action_primitives
    if not _import_action_primitives():
        raise ImportError("Action primitives could not be imported due to missing dependencies")
    
    # Import the specific classes we need
    from ..local_action_primitives.action_primitives import (
        Cutting, Slicing, Peeling, Coverage
    )

    # Handle point cloud input
    if isinstance(pcloud, str):
        pcloud = Pointcloud(filename=pcloud)
    elif not hasattr(pcloud, 'vertices'):
        raise TypeError("pcloud must be a Pointcloud object or filename string")

    # Define mapping of primitive types to classes
    PRIMITIVE_CLASSES = {
        'cutting': Cutting,
        'slicing': Slicing,
        'peeling': Peeling,
        'coverage': Coverage,
    }
    
    if primitive_type not in PRIMITIVE_CLASSES:
        raise ValueError(f"Unknown primitive type: {primitive_type}. "
                        f"Supported types: {list(PRIMITIVE_CLASSES.keys())}")
    
    controller_class = PRIMITIVE_CLASSES[primitive_type]
    
    # Prepare constructor arguments
    init_args = {
        'pcloud': pcloud,
        'primitive_type': primitive_type,
    }
    
    # Add optional arguments if provided
    if diffusion_scalar is not None:
        init_args['diffusion_scalar'] = diffusion_scalar
    if source_vertices is not None:
        init_args['source_vertices'] = source_vertices
    if start_vertex is not None:
        init_args['start_vertex'] = start_vertex
    if end_vertex is not None:
        init_args['end_vertex'] = end_vertex
        
    # Add any additional keyword arguments
    init_args.update(kwargs)
    
    # Create and return controller
    controller = controller_class(**init_args)
    
    print(f"✓ Created {primitive_type} controller with {len(pcloud.vertices)} vertices")
    
    return controller


def create_batch_controllers(
    primitive_type: str,
    pcloud: Union[Pointcloud, str],
    num_controllers: int,
    diffusion_scalars: Optional[np.ndarray] = None,
    **common_kwargs
) -> list:
    """
    Create multiple controllers for batch processing.
    
    Args:
        primitive_type: Type of primitive
        pcloud: Point cloud object or filename
        num_controllers: Number of controllers to create
        diffusion_scalars: Optional array of diffusion scalars
        **common_kwargs: Common arguments for all controllers
        
    Returns:
        List of initialized controllers
    """
    controllers = []
    
    # Handle point cloud input
    if isinstance(pcloud, str):
        pcloud = Pointcloud(filename=pcloud)
    
    # Generate default diffusion scalars if not provided
    if diffusion_scalars is None:
        diffusion_scalars = np.logspace(np.log10(0.1), np.log10(10000), num_controllers)
    
    for i in range(num_controllers):
        # Use specific diffusion scalar if array provided
        if i < len(diffusion_scalars):
            diffusion_scalar = diffusion_scalars[i]
        else:
            diffusion_scalar = None
        
        controller = create_primitive_controller(
            primitive_type=primitive_type,
            pcloud=pcloud,
            diffusion_scalar=diffusion_scalar,
            **common_kwargs
        )
        
        controllers.append(controller)
    
    print(f"✓ Created {len(controllers)} {primitive_type} controllers")
    
    return controllers


def create_controller_from_config(
    config: Dict[str, Any],
    pcloud: Optional[Union[Pointcloud, str]] = None
) -> Any:
    """
    Create controller from configuration dictionary.
    
    Args:
        config: Configuration dictionary with controller parameters
        pcloud: Optional point cloud (can be overridden by config)
        
    Returns:
        Initialized controller
        
    Example config:
        {
            'primitive_type': 'cutting',
            'filename': 'banana_half.ply',  # or use pcloud parameter
            'diffusion_scalar': 1000.0,
            'source_vertices': [10, 20],
            'start_vertex': 10
        }
    """
    config = config.copy()  # Don't modify original config
    
    # Extract primitive type (required)
    primitive_type = config.pop('primitive_type')
    
    # Handle point cloud source
    if 'filename' in config:
        pcloud = config.pop('filename')
    elif pcloud is None:
        raise ValueError("Either 'filename' must be in config or pcloud must be provided")
    
    # Create controller with remaining config as kwargs
    return create_primitive_controller(
        primitive_type=primitive_type,
        pcloud=pcloud,
        **config
    )


def get_primitive_defaults(primitive_type: str) -> Dict[str, Any]:
    """
    Get default parameters for a specific primitive type.

    Args:
        primitive_type: Type of primitive

    Returns:
        Dictionary of default parameters
    """
    defaults = {
        'cutting': {
            'diffusion_scalar': 1000.0,
        },
        'slicing': {
            'diffusion_scalar': 1000.0,
        },
        'peeling': {
            'diffusion_scalar': 1000.0,
        },
        'coverage': {
            'diffusion_scalar': 1000.0,
        },
    }

    return defaults.get(primitive_type, {})


def validate_primitive_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary for primitive creation.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_keys = ['primitive_type']
    
    # Check required keys
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")
    
    # Validate primitive type
    valid_types = ['cutting', 'slicing', 'peeling', 'coverage']
    if config['primitive_type'] not in valid_types:
        raise ValueError(f"Invalid primitive_type. Must be one of: {valid_types}")
    
    # Validate optional parameters
    if 'diffusion_scalar' in config:
        if not isinstance(config['diffusion_scalar'], (int, float)) or config['diffusion_scalar'] <= 0:
            raise ValueError("diffusion_scalar must be a positive number")
    
    if 'source_vertices' in config:
        source_vertices = config['source_vertices']
        if not isinstance(source_vertices, (list, np.ndarray)) or len(source_vertices) == 0:
            raise ValueError("source_vertices must be a non-empty list or array")
    
    if 'start_vertex' in config:
        if not isinstance(config['start_vertex'], int) or config['start_vertex'] < 0:
            raise ValueError("start_vertex must be a non-negative integer")
    
    return True


def create_experiment_suite(
    primitive_types: list,
    pcloud: Union[Pointcloud, str],
    **common_kwargs
) -> Dict[str, Any]:
    """
    Create a suite of controllers for different primitive types.
    
    Args:
        primitive_types: List of primitive type names
        pcloud: Point cloud object or filename
        **common_kwargs: Common arguments for all controllers
        
    Returns:
        Dictionary mapping primitive types to controllers
    """
    suite = {}
    
    # Handle point cloud input
    if isinstance(pcloud, str):
        pcloud = Pointcloud(filename=pcloud)
    
    for primitive_type in primitive_types:
        try:
            controller = create_primitive_controller(
                primitive_type=primitive_type,
                pcloud=pcloud,
                **common_kwargs
            )
            suite[primitive_type] = controller
        except Exception as e:
            print(f"Warning: Failed to create {primitive_type} controller: {e}")
    
    print(f"✓ Created experiment suite with {len(suite)} controllers")
    
    return suite