"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Noise generation utilities for experiments.

This module provides standardized noise generation functions to eliminate
duplicate code across batch processing scripts.
"""

import numpy as np
from typing import Optional, Tuple, Union


def generate_keypoint_noise(
    original_keypoints: np.ndarray,
    scale: float = 0.02,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate noise for keypoint positions.
    
    Args:
        original_keypoints: Original keypoint positions (N, 3)
        scale: Standard deviation of noise
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (noise, noisy_keypoints)
    """
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(scale=scale, size=original_keypoints.shape)
    noisy_keypoints = original_keypoints + noise
    
    return noise, noisy_keypoints


def generate_geometric_noise(
    vertices_shape: Tuple[int, int],
    scale: float = 0.003,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate geometric noise for point cloud vertices.
    
    Args:
        vertices_shape: Shape of the vertices array (N, 3)
        scale: Standard deviation of noise
        seed: Random seed for reproducibility
        
    Returns:
        Noise array with same shape as vertices
    """
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(scale=scale, size=vertices_shape)
    return noise


def generate_scaling_factors(
    num_factors: int,
    low: float = 0.6,
    high: float = 1.4,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random scaling factors.
    
    Args:
        num_factors: Number of scaling factors to generate
        low: Minimum scaling factor
        high: Maximum scaling factor  
        seed: Random seed for reproducibility
        
    Returns:
        Array of scaling factors
    """
    if seed is not None:
        np.random.seed(seed)
    
    scaling_factors = np.random.uniform(low, high, size=num_factors)
    return scaling_factors


def generate_topological_noise(
    vertices: np.ndarray,
    noise_type: str = "gaussian",
    **noise_params
) -> np.ndarray:
    """
    Generate topological noise for point clouds.
    
    Args:
        vertices: Original vertices (N, 3)
        noise_type: Type of noise ('gaussian', 'uniform', 'bend', 'bulge', 'twist')
        **noise_params: Parameters specific to noise type
        
    Returns:
        Noisy vertices
    """
    if noise_type == "gaussian":
        scale = noise_params.get("scale", 0.01)
        seed = noise_params.get("seed")
        noise = generate_geometric_noise(vertices.shape, scale, seed)
        return vertices + noise
    
    elif noise_type == "uniform":
        scale = noise_params.get("scale", 0.01)
        seed = noise_params.get("seed")
        if seed is not None:
            np.random.seed(seed)
        noise = np.random.uniform(-scale, scale, size=vertices.shape)
        return vertices + noise
    
    elif noise_type == "bend":
        return _apply_bend_noise(vertices, **noise_params)
    
    elif noise_type == "bulge":
        return _apply_bulge_noise(vertices, **noise_params)
    
    elif noise_type == "twist":
        return _apply_twist_noise(vertices, **noise_params)
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def _apply_bend_noise(
    points: np.ndarray, 
    bend_axis: int = 2, 
    curvature: float = 0.01
) -> np.ndarray:
    """
    Apply bending transformation to points.
    
    Args:
        points: Input points (N, 3)
        bend_axis: Axis to bend along (0=x, 1=y, 2=z)
        curvature: Amount of curvature
        
    Returns:
        Bent points
    """
    bent = points.copy()
    other_axes = [i for i in range(3) if i != bend_axis]
    
    coord = bent[:, bend_axis]
    for ax in other_axes:
        bent[:, ax] += curvature * coord**2  # quadratic bend
    
    return bent


def _apply_bulge_noise(
    points: np.ndarray,
    amount: float = 0.05,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Apply bulging transformation to points.
    
    Args:
        points: Input points (N, 3)
        amount: Maximum bulge amount
        seed: Random seed for reproducibility
        
    Returns:
        Bulged points
    """
    if seed is not None:
        np.random.seed(seed)
    
    center = points.mean(axis=0)
    directions = points - center
    norm = np.linalg.norm(directions, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    directions /= norm
    
    r = np.random.uniform(-amount, amount, size=(points.shape[0], 1))
    return points + directions * r


def _apply_twist_noise(
    points: np.ndarray,
    axis: int = 2,
    twist_strength: float = 2.0
) -> np.ndarray:
    """
    Apply twisting transformation to points.
    
    Args:
        points: Input points (N, 3)
        axis: Axis to twist around (0=x, 1=y, 2=z)
        twist_strength: Strength of twist
        
    Returns:
        Twisted points
    """
    twisted = points.copy()
    z = twisted[:, axis]
    angle = twist_strength * z  # twist amount depends on coordinate
    
    # Apply rotation in the plane perpendicular to twist axis
    if axis == 2:  # twist around z-axis
        x, y = twisted[:, 0], twisted[:, 1]
        twisted[:, 0] = x * np.cos(angle) - y * np.sin(angle)
        twisted[:, 1] = x * np.sin(angle) + y * np.cos(angle)
    elif axis == 1:  # twist around y-axis
        x, z = twisted[:, 0], twisted[:, 2]
        twisted[:, 0] = x * np.cos(angle) - z * np.sin(angle)
        twisted[:, 2] = x * np.sin(angle) + z * np.cos(angle)
    elif axis == 0:  # twist around x-axis
        y, z = twisted[:, 1], twisted[:, 2]
        twisted[:, 1] = y * np.cos(angle) - z * np.sin(angle)
        twisted[:, 2] = y * np.sin(angle) + z * np.cos(angle)
    
    return twisted


def generate_batch_noise(
    noise_configs: list,
    vertices: np.ndarray,
    seed_base: int = 42
) -> list:
    """
    Generate multiple noise realizations for batch processing.
    
    Args:
        noise_configs: List of noise configuration dictionaries
        vertices: Original vertices to apply noise to
        seed_base: Base seed for reproducible noise generation
        
    Returns:
        List of noisy vertex arrays
    """
    noisy_batches = []
    
    for i, config in enumerate(noise_configs):
        config = config.copy()  # Don't modify original config
        config['seed'] = seed_base + i
        
        noise_type = config.pop('type', 'gaussian')
        noisy_vertices = generate_topological_noise(vertices, noise_type, **config)
        noisy_batches.append(noisy_vertices)
    
    return noisy_batches