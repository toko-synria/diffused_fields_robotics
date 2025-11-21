"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Coordinate system utilities for robotics applications.

This module provides coordinate system transformations and utilities
that are commonly used across peeling and other manipulation tasks.
"""

import numpy as np
from typing import Tuple, Optional


def compute_body_fixed_coordinate_system(
    source_vertices: np.ndarray,
    pcloud_vertices: np.ndarray
) -> np.ndarray:
    """
    Compute body-fixed coordinate system from source vertices.

    Body-fixed frame: A single fixed coordinate system whose x-axis is aligned
    with the line connecting the two source vertices. This frame doesn't change
    along the trajectory.

    This function implements the complex coordinate transformation logic
    that was duplicated across multiple peeling scripts.

    Args:
        source_vertices: Array of source vertex indices [start_idx, end_idx]
        pcloud_vertices: Point cloud vertices array (N, 3)

    Returns:
        Rotation matrix R (3x3) representing the body-fixed coordinate system
    """
    if len(source_vertices) < 2:
        raise ValueError("Need at least 2 source vertices to compute coordinate system")
    
    # Step 1: Get the new x-axis from source vertices
    start_point = pcloud_vertices[source_vertices[0]]
    end_point = pcloud_vertices[source_vertices[1]]
    x_axis = end_point - start_point
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Step 2: Choose an arbitrary reference vector
    ref = np.array([0, 0, 1])  # Default reference: z-axis
    
    # If x_axis is parallel to reference, choose different reference
    if np.allclose(np.abs(np.dot(x_axis, ref)), 1.0):
        ref = np.array([0, 1, 0])  # Use y-axis instead
    
    # Step 3: Compute orthogonal y and z axes using Gram-Schmidt
    z_axis = np.cross(x_axis, ref)
    z_axis /= np.linalg.norm(z_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    
    # Step 4: Create rotation matrix
    body_fixed_frame_R = np.stack([x_axis, y_axis, z_axis], axis=1)

    return body_fixed_frame_R


def apply_coordinate_transformation(
    points: np.ndarray,
    rotation_matrix: np.ndarray,
    translation: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply coordinate transformation to points.
    
    Args:
        points: Input points (N, 3)
        rotation_matrix: 3x3 rotation matrix
        translation: Optional translation vector (3,)
        
    Returns:
        Transformed points (N, 3)
    """
    # Apply rotation
    transformed_points = points @ rotation_matrix.T
    
    # Apply translation if provided
    if translation is not None:
        transformed_points += translation
    
    return transformed_points


def compute_orthogonal_basis(
    primary_vector: np.ndarray,
    reference_vector: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute orthogonal basis from a primary vector.
    
    Args:
        primary_vector: Primary direction vector (3,)
        reference_vector: Optional reference for second axis (3,)
        
    Returns:
        Orthogonal basis matrix (3, 3) where columns are basis vectors
    """
    # Normalize primary vector
    v1 = primary_vector / np.linalg.norm(primary_vector)
    
    # Choose reference vector if not provided
    if reference_vector is None:
        # Use z-axis, or y-axis if primary is parallel to z
        if np.abs(np.dot(v1, [0, 0, 1])) > 0.9:
            reference_vector = np.array([0, 1, 0])
        else:
            reference_vector = np.array([0, 0, 1])
    
    # Gram-Schmidt orthogonalization
    v2 = reference_vector - np.dot(reference_vector, v1) * v1
    v2 = v2 / np.linalg.norm(v2)
    
    # Third vector from cross product
    v3 = np.cross(v1, v2)
    v3 = v3 / np.linalg.norm(v3)
    
    return np.column_stack([v1, v2, v3])


def transform_to_local_coordinates(
    global_points: np.ndarray,
    origin: np.ndarray,
    local_basis: np.ndarray
) -> np.ndarray:
    """
    Transform points from global to local coordinate system.
    
    Args:
        global_points: Points in global coordinates (N, 3)
        origin: Origin of local coordinate system (3,)
        local_basis: Local coordinate basis (3, 3)
        
    Returns:
        Points in local coordinates (N, 3)
    """
    # Translate to local origin
    translated_points = global_points - origin
    
    # Rotate to local basis (inverse rotation)
    local_points = translated_points @ local_basis
    
    return local_points


def transform_to_global_coordinates(
    local_points: np.ndarray,
    origin: np.ndarray,
    local_basis: np.ndarray
) -> np.ndarray:
    """
    Transform points from local to global coordinate system.
    
    Args:
        local_points: Points in local coordinates (N, 3)
        origin: Origin of local coordinate system (3,)
        local_basis: Local coordinate basis (3, 3)
        
    Returns:
        Points in global coordinates (N, 3)
    """
    # Rotate to global basis
    global_points = local_points @ local_basis.T
    
    # Translate to global origin
    global_points += origin
    
    return global_points


def compute_trajectory_tangents(
    trajectory: np.ndarray,
    smoothing_window: int = 3
) -> np.ndarray:
    """
    Compute tangent vectors along a trajectory.
    
    Args:
        trajectory: Trajectory points (N, 3)
        smoothing_window: Window size for smoothing tangents
        
    Returns:
        Unit tangent vectors (N, 3)
    """
    if len(trajectory) < 2:
        raise ValueError("Need at least 2 points to compute tangents")
    
    # Ensure we're working with float arrays
    trajectory = np.array(trajectory, dtype=np.float64)
    tangents = np.zeros_like(trajectory, dtype=np.float64)
    
    # Forward differences at start
    tangents[0] = trajectory[1] - trajectory[0]
    
    # Central differences in middle
    for i in range(1, len(trajectory) - 1):
        tangents[i] = trajectory[i + 1] - trajectory[i - 1]
    
    # Backward differences at end
    tangents[-1] = trajectory[-1] - trajectory[-2]
    
    # Normalize tangent vectors
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    tangents = tangents / norms
    
    # Optional smoothing
    if smoothing_window > 1 and len(trajectory) > smoothing_window:
        tangents = _smooth_vectors(tangents, smoothing_window)
    
    return tangents


def compute_trajectory_normals(
    trajectory: np.ndarray,
    tangents: Optional[np.ndarray] = None,
    up_vector: np.ndarray = np.array([0, 0, 1])
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normal and binormal vectors along a trajectory.
    
    Args:
        trajectory: Trajectory points (N, 3)
        tangents: Pre-computed tangent vectors (N, 3)
        up_vector: Global up vector for reference (3,)
        
    Returns:
        Tuple of (normal_vectors, binormal_vectors) each (N, 3)
    """
    if tangents is None:
        tangents = compute_trajectory_tangents(trajectory)
    
    normals = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)
    
    for i, tangent in enumerate(tangents):
        # Compute normal as cross product with up vector
        normal = np.cross(up_vector, tangent)
        normal_norm = np.linalg.norm(normal)
        
        if normal_norm > 1e-6:
            normal /= normal_norm
        else:
            # Tangent is parallel to up vector, choose perpendicular
            if np.abs(tangent[0]) < 0.9:
                normal = np.array([1, 0, 0])
            else:
                normal = np.array([0, 1, 0])
            normal = normal - np.dot(normal, tangent) * tangent
            normal /= np.linalg.norm(normal)
        
        # Binormal from cross product
        binormal = np.cross(tangent, normal)
        
        normals[i] = normal
        binormals[i] = binormal
    
    return normals, binormals


def _smooth_vectors(vectors: np.ndarray, window_size: int) -> np.ndarray:
    """
    Smooth vectors using a moving average window.
    
    Args:
        vectors: Input vectors (N, 3)
        window_size: Size of smoothing window
        
    Returns:
        Smoothed vectors (N, 3)
    """
    smoothed = vectors.copy()
    half_window = window_size // 2
    
    for i in range(len(vectors)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(vectors), i + half_window + 1)
        
        # Average vectors in window
        smoothed[i] = np.mean(vectors[start_idx:end_idx], axis=0)
        
        # Renormalize
        norm = np.linalg.norm(smoothed[i])
        if norm > 1e-6:
            smoothed[i] /= norm
    
    return smoothed