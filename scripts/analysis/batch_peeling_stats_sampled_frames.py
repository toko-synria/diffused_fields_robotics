"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Batch peeling statistics using sampled local frames as discrete approximation.

This script samples K frames from the continuous local frame field (pcloud.local_bases)
and uses them as multiple body-fixed reference frames with distance-based blending.
This creates a discrete approximation of using fully continuous local frames.

Compares:
1. Single body-fixed frame
2. Multi-frame (K sampled frames from local field)
3. Fully continuous local frames

By varying K, we can analyze the trade-off between number of frames and performance.
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from diffused_fields.visualization.plotting_ps import plot_orientation_field

# Import plots directory configuration
from diffused_fields_robotics.core.config import get_plots_dir
from diffused_fields_robotics.utils import (
    align_by_transitions,
    get_ground_truth_transitions,
    load_results,
)


def sample_frames_from_pointcloud(
    pointcloud_vertices: np.ndarray,
    pointcloud_local_bases: np.ndarray,
    frame_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract K frames from pointcloud at specified indices.

    Args:
        pointcloud_vertices: Pointcloud vertices (M, 3)
        pointcloud_local_bases: Pointcloud local bases (M, 3, 3)
        frame_indices: Indices to sample (K,)

    Returns:
        frame_positions: (K, 3) sampled positions
        frame_orientations: (K, 3, 3) sampled local bases
    """
    frame_positions = pointcloud_vertices[frame_indices]
    frame_orientations = pointcloud_local_bases[frame_indices]
    return frame_positions, frame_orientations


def farthest_point_sampling(
    points: np.ndarray, num_samples: int, seed_indices: np.ndarray = None
) -> np.ndarray:
    """
    Farthest point sampling for diverse frame selection.

    Args:
        points: Point cloud (N, 3)
        num_samples: Number of points to sample
        seed_indices: Optional seed indices to start FPS from (e.g., keypoints)

    Returns:
        indices: Selected point indices
    """
    N = len(points)
    indices = np.zeros(num_samples, dtype=int)
    distances = np.full(N, np.inf)

    # Start with seed points if provided
    if seed_indices is not None and len(seed_indices) > 0:
        num_seeds = min(len(seed_indices), num_samples)
        indices[:num_seeds] = seed_indices[:num_seeds]

        # Initialize distances from seed points
        for seed_idx in indices[:num_seeds]:
            dist_to_seed = np.linalg.norm(points - points[seed_idx], axis=1)
            distances = np.minimum(distances, dist_to_seed)

        start_idx = num_seeds
    else:
        # Start with first point if no seeds
        indices[0] = 0
        start_idx = 1

    # Continue FPS from seed points
    for i in range(start_idx, num_samples):
        # Select farthest point from all selected points so far
        indices[i] = np.argmax(distances)

        # Update distances
        last_idx = indices[i]
        dist_to_last = np.linalg.norm(points - points[last_idx], axis=1)
        distances = np.minimum(distances, dist_to_last)

    return indices


def compute_frame_weights(
    query_points: np.ndarray, frame_positions: np.ndarray, temperature: float = 0.001
) -> np.ndarray:
    """
    Compute distance-based weights for frame blending using softmax.

    Args:
        query_points: Query positions (N, 3)
        frame_positions: Frame positions (K, 3)
        temperature: Temperature for softmax (smaller = sharper transitions)

    Returns:
        weights: (N, K) blending weights
    """
    N = len(query_points)
    K = len(frame_positions)
    weights = np.zeros((N, K))

    for i, point in enumerate(query_points):
        # Compute distances to all frames
        distances = np.linalg.norm(frame_positions - point[np.newaxis, :], axis=1)

        # Softmax with temperature
        exp_vals = np.exp(-distances / temperature)
        weights[i] = exp_vals / np.sum(exp_vals)

    return weights


def compute_multiframe_velocities(
    trajectory: np.ndarray,
    pointcloud_vertices: np.ndarray,
    pointcloud_local_bases: np.ndarray,
    frame_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute velocities using K sampled frames from pointcloud with blending.

    Args:
        trajectory: Trajectory points (N, 3)
        pointcloud_vertices: Pointcloud vertices (M, 3)
        pointcloud_local_bases: Pointcloud local bases (M, 3, 3)
        frame_indices: Indices of frames to use (K,)

    Returns:
        velocities_multiframe: Blended velocities (N-1, 3)
        frame_weights: Blending weights (N-1, K)
        frame_positions: Sampled frame positions (K, 3)
    """
    # Extract frames at specified indices
    frame_positions, frame_orientations = sample_frames_from_pointcloud(
        pointcloud_vertices, pointcloud_local_bases, frame_indices
    )

    # Compute velocities using the sampled frames
    velocities_multiframe = compute_multiframe_velocities_with_frames(
        trajectory, frame_positions, frame_orientations
    )

    # Compute weights for diagnostics
    frame_weights = compute_frame_weights(trajectory[:-1], frame_positions)

    return velocities_multiframe, frame_weights, frame_positions


def compute_raw_multiframe_data(
    trajectory: np.ndarray,
    frame_positions: np.ndarray,
    frame_orientations: np.ndarray,
) -> dict:
    """
    Compute raw data for multiframe velocity computation (temperature-independent).

    This function extracts all the data needed to later apply different temperature
    values for blending without recomputing everything.

    Args:
        trajectory: Trajectory points (N+1, 3)
        frame_positions: Sampled frame positions (K, 3)
        frame_orientations: Sampled frame orientations (K, 3, 3)

    Returns:
        Dictionary containing:
            - trajectory_velocity_points: Points where velocities are defined (N, 3)
            - velocities_global: Global frame velocities (N, 3)
            - frame_positions: Sampled frame positions (K, 3)
            - frame_orientations: Sampled frame orientations (K, 3, 3)
            - velocities_in_frames: Velocity transformed to each frame (N, K, 3)
            - nearest_frame_indices: Index of nearest frame for each point (N,)
    """
    # Compute velocities (N velocity vectors from N+1 trajectory points)
    velocities_global = np.diff(trajectory, axis=0)
    N = len(velocities_global)
    K = len(frame_positions)

    # Get trajectory points where velocities are defined (first N points)
    trajectory_velocity_points = trajectory[:-1]

    # Transform each velocity to all frames (N x K x 3)
    velocities_in_frames = np.zeros((N, K, 3))
    for i in range(N):
        for k in range(K):
            velocities_in_frames[i, k] = frame_orientations[k].T @ velocities_global[i]

    # Find nearest frame index for each trajectory point
    nearest_frame_indices = np.zeros(N, dtype=int)
    for i in range(N):
        distances = np.linalg.norm(
            frame_positions - trajectory_velocity_points[i], axis=1
        )
        nearest_frame_indices[i] = np.argmin(distances)

    return {
        "trajectory_velocity_points": trajectory_velocity_points,
        "velocities_global": velocities_global,
        "frame_positions": frame_positions,
        "frame_orientations": frame_orientations,
        "velocities_in_frames": velocities_in_frames,
        "nearest_frame_indices": nearest_frame_indices,
    }


def apply_temperature_blending(
    raw_data: dict,
    temperature: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply temperature-dependent blending to precomputed raw data.

    Args:
        raw_data: Dictionary from compute_raw_multiframe_data
        temperature: Temperature for softmax weighting

    Returns:
        velocities_blended: Blended velocities (N, 3)
        velocities_nearest: Velocities in nearest frame (N, 3)
    """
    trajectory_velocity_points = raw_data["trajectory_velocity_points"]
    frame_positions = raw_data["frame_positions"]
    velocities_in_frames = raw_data["velocities_in_frames"]
    nearest_frame_indices = raw_data["nearest_frame_indices"]

    N, K, _ = velocities_in_frames.shape

    # Compute weights for velocity points based on their positions
    frame_weights = compute_frame_weights(
        trajectory_velocity_points, frame_positions, temperature
    )

    # Blend velocities using weights
    velocities_blended = np.zeros((N, 3))
    for i in range(N):
        for k in range(K):
            velocities_blended[i] += frame_weights[i, k] * velocities_in_frames[i, k]

    # Extract nearest frame velocities
    velocities_nearest = velocities_in_frames[np.arange(N), nearest_frame_indices]

    return velocities_blended, velocities_nearest


def compute_multiframe_velocities_with_frames(
    trajectory: np.ndarray,
    frame_positions: np.ndarray,
    frame_orientations: np.ndarray,
    temperature: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocities using pre-sampled frames with blending and nearest frame.

    Args:
        trajectory: Trajectory points (N+1, 3)
        frame_positions: Sampled frame positions (K, 3)
        frame_orientations: Sampled frame orientations (K, 3, 3)
        temperature: Temperature for softmax weighting

    Returns:
        velocities_blended: Blended velocities (N, 3)
        velocities_nearest: Velocities in nearest frame (N, 3)
    """
    # Compute raw data
    raw_data = compute_raw_multiframe_data(
        trajectory, frame_positions, frame_orientations
    )

    # Apply temperature-dependent blending
    return apply_temperature_blending(raw_data, temperature)


def compute_stats_from_raw_data(
    raw_data_all: list,
    transitions_list: List[np.ndarray],
    temperature: float = 0.001,
) -> dict:
    """
    Compute statistics from cached raw data using a specific temperature.

    Args:
        raw_data_all: List of raw data dictionaries for all experiments
        transitions_list: Ground truth transition indices for each experiment
        temperature: Temperature for softmax blending

    Returns:
        Dictionary with aligned velocities and statistics
    """
    velocities_blended_all = []
    velocities_nearest_all = []

    # Apply temperature-dependent blending to each experiment
    for raw_data in raw_data_all:
        vel_blended, vel_nearest = apply_temperature_blending(raw_data, temperature)
        velocities_blended_all.append(vel_blended)
        velocities_nearest_all.append(vel_nearest)

    # Align both blended and nearest
    velocities_blended_aligned, ref_idx_align = align_by_transitions(
        velocities_blended_all, transitions_list
    )
    velocities_nearest_aligned, _ = align_by_transitions(
        velocities_nearest_all, transitions_list
    )

    # Return statistics
    return {
        "velocities_blended": velocities_blended_aligned,
        "velocities_nearest": velocities_nearest_aligned,
        "mean_blended": velocities_blended_aligned.mean(axis=0),
        "std_blended": velocities_blended_aligned.std(axis=0),
        "mean_nearest": velocities_nearest_aligned.mean(axis=0),
        "std_nearest": velocities_nearest_aligned.std(axis=0),
        "max_len": velocities_blended_aligned.shape[1],
        "num_experiments": len(velocities_blended_aligned),
        "transitions": transitions_list,
        "ref_index": ref_idx_align,
    }


def compute_velocity_stats_single_frame(
    results: list, transitions_list: List[np.ndarray], valid_indices: List[int]
) -> dict:
    """Compute statistics using single body-fixed frame."""
    velocities_all = []
    for idx in valid_indices:
        velocities_all.append(results[idx]["velocities_body_fixed"])

    velocities_aligned, ref_idx = align_by_transitions(velocities_all, transitions_list)

    print(f"Single Body-Fixed: Aligned {len(velocities_all)} demos")

    return {
        "velocities": velocities_aligned,
        "mean": velocities_aligned.mean(axis=0),
        "std": velocities_aligned.std(axis=0),
        "max_len": velocities_aligned.shape[1],
        "num_experiments": len(velocities_aligned),
        "transitions": transitions_list,
        "ref_index": ref_idx,
    }


def compute_velocity_stats_multiframe(
    results: list,
    transitions_list: List[np.ndarray],
    valid_indices: List[int],
    num_frames: int = 5,
) -> dict:
    """Compute statistics using K sampled frames with FPS seeded from keypoints."""

    # Sample K frame indices ONCE from reference pointcloud (first valid experiment)
    ref_idx = valid_indices[0]
    ref_result = results[ref_idx]
    ref_pointcloud_vertices = ref_result["pointcloud"]["vertices"]

    # Get source vertices (keypoints) to seed FPS - ensures alignment with body-fixed frame
    source_vertices = ref_result["source_vertices"]

    # Sample frames using FPS, starting from keypoints for proper alignment
    frame_indices = farthest_point_sampling(
        ref_pointcloud_vertices, num_frames, seed_indices=source_vertices
    )

    print(f"Sampled {num_frames} frames from reference pointcloud using FPS")
    print(f"  Seed indices (keypoints): {source_vertices}")
    print(f"  Sampled frame indices: {frame_indices}")

    # Now use these SAME indices for all experiments
    velocities_blended_all = []
    velocities_nearest_all = []

    for idx in valid_indices:
        result = results[idx]
        trajectory = np.array(result["trajectory"])
        pointcloud_vertices = result["pointcloud"]["vertices"]
        pointcloud_local_bases = result["pointcloud"]["local_bases"]

        # Extract frames at the SAME pointcloud indices for this experiment
        frame_positions = pointcloud_vertices[frame_indices]
        frame_orientations = pointcloud_local_bases[frame_indices]

        # Compute multi-frame velocities using these frames
        vel_blended, vel_nearest = compute_multiframe_velocities_with_frames(
            trajectory,
            frame_positions,
            frame_orientations,
        )
        velocities_blended_all.append(vel_blended)
        velocities_nearest_all.append(vel_nearest)

    # Align both blended and nearest
    velocities_blended_aligned, ref_idx = align_by_transitions(
        velocities_blended_all, transitions_list
    )
    velocities_nearest_aligned, _ = align_by_transitions(
        velocities_nearest_all, transitions_list
    )

    print(
        f"Multi-Frame (K={num_frames}, FPS): Aligned {len(velocities_blended_all)} demos"
    )

    return {
        "velocities_blended": velocities_blended_aligned,
        "velocities_nearest": velocities_nearest_aligned,
        "mean_blended": velocities_blended_aligned.mean(axis=0),
        "std_blended": velocities_blended_aligned.std(axis=0),
        "mean_nearest": velocities_nearest_aligned.mean(axis=0),
        "std_nearest": velocities_nearest_aligned.std(axis=0),
        "max_len": velocities_blended_aligned.shape[1],
        "num_experiments": len(velocities_blended_aligned),
        "num_frames": num_frames,
        "transitions": transitions_list,
        "ref_index": ref_idx,
        "frame_indices": frame_indices,
    }


def compute_velocity_stats_local(
    results: list, transitions_list: List[np.ndarray], valid_indices: List[int]
) -> dict:
    """Compute statistics using fully continuous local frames."""
    velocities_all = []
    for idx in valid_indices:
        velocities_all.append(results[idx]["velocities_local"])

    velocities_aligned, ref_idx = align_by_transitions(velocities_all, transitions_list)

    print(f"Local Frames (Continuous): Aligned {len(velocities_all)} demos")

    return {
        "velocities": velocities_aligned,
        "mean": velocities_aligned.mean(axis=0),
        "std": velocities_aligned.std(axis=0),
        "max_len": velocities_aligned.shape[1],
        "num_experiments": len(velocities_aligned),
        "transitions": transitions_list,
        "ref_index": ref_idx,
    }


def plot_comparison_multiple_K(
    results: list,
    K_values: List[int],
    save_path: str = None,
    precomputed_stats: dict = None,
) -> plt.Figure:
    """
    Plot comparison for multiple values of K (number of sampled frames).

    Args:
        results: Experiment results
        K_values: List of K values to test
        save_path: Path to save plot
        precomputed_stats: Optional precomputed statistics to avoid recomputation

    Returns:
        Figure object
    """
    if precomputed_stats is None:
        transitions_list, valid_indices = get_ground_truth_transitions(results)

        if not transitions_list:
            print("Error: No ground truth transitions found")
            return None

        # Compute statistics for single frame and local
        single_stats = compute_velocity_stats_single_frame(
            results, transitions_list, valid_indices
        )
        local_stats = compute_velocity_stats_local(
            results, transitions_list, valid_indices
        )

        # Compute statistics for each K value
        multiframe_stats_list = []
        for K in K_values:
            stats = compute_velocity_stats_multiframe(
                results, transitions_list, valid_indices, K
            )
            multiframe_stats_list.append(stats)
    else:
        # Use precomputed statistics
        single_stats = precomputed_stats["single"]
        local_stats = precomputed_stats["local"]
        all_K_values = precomputed_stats["K_values"]
        all_multiframe_stats = precomputed_stats["multiframe"]

        # Filter to only the requested K values
        multiframe_stats_list = [
            all_multiframe_stats[all_K_values.index(K)]
            for K in K_values
            if K in all_K_values
        ]

    # Setup plot: 3 rows (X, Y, Z) x (2 + len(K_values)) columns
    num_cols = 2 + len(K_values)
    fig, axes = plt.subplots(3, num_cols, figsize=(5 * num_cols, 8), sharex=True)

    axis_labels = ["X", "Y", "Z"]
    colors = ["red", "green", "blue"]

    # Column 0: Single body-fixed
    for i in range(3):
        ax = axes[i, 0]
        ax.plot(single_stats["mean"][:, i], color=colors[i], linewidth=2)
        ax.fill_between(
            np.arange(single_stats["max_len"]),
            single_stats["mean"][:, i] - single_stats["std"][:, i],
            single_stats["mean"][:, i] + single_stats["std"][:, i],
            color=colors[i],
            alpha=0.3,
        )
        ax.axhline(0, linestyle="--", color="gray", linewidth=1)
        ax.set_ylim(-0.0015, 0.0015)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_title("Single\nBody-Fixed (K=1)", fontsize=15)
        if i == 0:
            ax.set_ylabel(f"{axis_labels[i]} Velocity", fontsize=15)
        else:
            ax.set_ylabel(axis_labels[i], fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=15)

    # Columns 1 to len(K_values): Multi-frame with different K (using blended)
    for col_idx, (K, stats) in enumerate(zip(K_values, multiframe_stats_list), start=1):
        for i in range(3):
            ax = axes[i, col_idx]
            ax.plot(
                stats["mean_blended"][:, i],
                color=colors[i],
                linewidth=2,
                label="Blended",
            )
            ax.fill_between(
                np.arange(stats["max_len"]),
                stats["mean_blended"][:, i] - stats["std_blended"][:, i],
                stats["mean_blended"][:, i] + stats["std_blended"][:, i],
                color=colors[i],
                alpha=0.3,
            )
            ax.axhline(0, linestyle="--", color="gray", linewidth=1)
            ax.set_ylim(-0.0015, 0.0015)
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.set_title(f"Sampled\nFrames (K={K})\nBlended", fontsize=15)
            ax.tick_params(axis="both", which="major", labelsize=15)

    # Last column: Local frames
    for i in range(3):
        ax = axes[i, -1]
        ax.plot(local_stats["mean"][:, i], color=colors[i], linewidth=2)
        ax.fill_between(
            np.arange(local_stats["max_len"]),
            local_stats["mean"][:, i] - local_stats["std"][:, i],
            local_stats["mean"][:, i] + local_stats["std"][:, i],
            color=colors[i],
            alpha=0.3,
        )
        ax.axhline(0, linestyle="--", color="gray", linewidth=1)
        ax.set_ylim(-0.0015, 0.0015)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_title("Continuous\nLocal Frames", fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=15)

    # X-axis labels
    for col in range(num_cols):
        axes[2, col].set_xlabel("Time Step", fontsize=15)

    plt.suptitle(
        f"Discrete Frame Approximation Analysis ({single_stats['num_experiments']} experiments)",
        fontsize=15,
        y=0.995,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved comparison plot to {save_path}")

    return fig


def plot_comparison_nearest_frame(
    results: list,
    K_values: List[int],
    save_path: str = None,
    precomputed_stats: dict = None,
) -> plt.Figure:
    """
    Plot comparison for multiple values of K using nearest frame (no blending).

    Args:
        results: Experiment results
        K_values: List of K values to test
        save_path: Path to save plot
        precomputed_stats: Optional precomputed statistics to avoid recomputation

    Returns:
        Figure object
    """
    if precomputed_stats is None:
        transitions_list, valid_indices = get_ground_truth_transitions(results)

        if not transitions_list:
            print("Error: No ground truth transitions found")
            return None

        # Compute statistics for single frame and local
        single_stats = compute_velocity_stats_single_frame(
            results, transitions_list, valid_indices
        )
        local_stats = compute_velocity_stats_local(
            results, transitions_list, valid_indices
        )

        # Compute statistics for each K value
        multiframe_stats_list = []
        for K in K_values:
            stats = compute_velocity_stats_multiframe(
                results, transitions_list, valid_indices, K
            )
            multiframe_stats_list.append(stats)
    else:
        # Use precomputed statistics
        single_stats = precomputed_stats["single"]
        local_stats = precomputed_stats["local"]
        all_K_values = precomputed_stats["K_values"]
        all_multiframe_stats = precomputed_stats["multiframe"]

        # Filter to only the requested K values
        multiframe_stats_list = [
            all_multiframe_stats[all_K_values.index(K)]
            for K in K_values
            if K in all_K_values
        ]

    # Setup plot: 3 rows (X, Y, Z) x (2 + len(K_values)) columns
    num_cols = 2 + len(K_values)
    fig, axes = plt.subplots(3, num_cols, figsize=(5 * num_cols, 8), sharex=True)

    axis_labels = ["X", "Y", "Z"]
    colors = ["red", "green", "blue"]

    # Column 0: Single body-fixed
    for i in range(3):
        ax = axes[i, 0]
        ax.plot(single_stats["mean"][:, i], color=colors[i], linewidth=2)
        ax.fill_between(
            np.arange(single_stats["max_len"]),
            single_stats["mean"][:, i] - single_stats["std"][:, i],
            single_stats["mean"][:, i] + single_stats["std"][:, i],
            color=colors[i],
            alpha=0.3,
        )
        ax.axhline(0, linestyle="--", color="gray", linewidth=1)
        ax.set_ylim(-0.0015, 0.0015)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_title("Single\nBody-Fixed (K=1)", fontsize=15)
        if i == 0:
            ax.set_ylabel(f"{axis_labels[i]} Velocity", fontsize=15)
        else:
            ax.set_ylabel(axis_labels[i], fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=15)

    # Columns 1 to len(K_values): Multi-frame with different K (using nearest)
    for col_idx, (K, stats) in enumerate(zip(K_values, multiframe_stats_list), start=1):
        for i in range(3):
            ax = axes[i, col_idx]
            ax.plot(
                stats["mean_nearest"][:, i],
                color=colors[i],
                linewidth=2,
                label="Nearest",
            )
            ax.fill_between(
                np.arange(stats["max_len"]),
                stats["mean_nearest"][:, i] - stats["std_nearest"][:, i],
                stats["mean_nearest"][:, i] + stats["std_nearest"][:, i],
                color=colors[i],
                alpha=0.3,
            )
            ax.axhline(0, linestyle="--", color="gray", linewidth=1)
            ax.set_ylim(-0.0015, 0.0015)
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.set_title(f"Sampled\nFrames (K={K})\nNearest", fontsize=15)
            ax.tick_params(axis="both", which="major", labelsize=15)

    # Last column: Local frames
    for i in range(3):
        ax = axes[i, -1]
        ax.plot(local_stats["mean"][:, i], color=colors[i], linewidth=2)
        ax.fill_between(
            np.arange(local_stats["max_len"]),
            local_stats["mean"][:, i] - local_stats["std"][:, i],
            local_stats["mean"][:, i] + local_stats["std"][:, i],
            color=colors[i],
            alpha=0.3,
        )
        ax.axhline(0, linestyle="--", color="gray", linewidth=1)
        ax.set_ylim(-0.0015, 0.0015)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_title("Continuous\nLocal Frames", fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=15)

    # X-axis labels
    for col in range(num_cols):
        axes[2, col].set_xlabel("Time Step", fontsize=15)

    plt.suptitle(
        f"Nearest Frame Analysis ({single_stats['num_experiments']} experiments)",
        fontsize=15,
        y=0.995,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved nearest frame comparison plot to {save_path}")

    return fig


def plot_variance_vs_K(
    results: list,
    K_range: List[int],
    save_path: str = None,
    precomputed_stats: dict = None,
) -> plt.Figure:
    """
    Plot variance as a function of K (number of sampled frames).

    Args:
        results: Experiment results
        K_range: Range of K values to test
        save_path: Path to save plot
        precomputed_stats: Optional precomputed statistics to avoid recomputation

    Returns:
        Figure object
    """
    if precomputed_stats is None:
        transitions_list, valid_indices = get_ground_truth_transitions(results)

        if not transitions_list:
            print("Error: No ground truth transitions found")
            return None

        # Compute baseline statistics
        single_stats = compute_velocity_stats_single_frame(
            results, transitions_list, valid_indices
        )
        local_stats = compute_velocity_stats_local(
            results, transitions_list, valid_indices
        )

        # Compute statistics for each K
        std_devs_blended = []
        std_devs_nearest = []
        for K in K_range:
            stats = compute_velocity_stats_multiframe(
                results, transitions_list, valid_indices, K
            )
            # Average standard deviation across all components and time (convert to mm/s)
            avg_std_blended = np.mean(stats["std_blended"]) * 1000.0
            avg_std_nearest = np.mean(stats["std_nearest"]) * 1000.0
            std_devs_blended.append(avg_std_blended)
            std_devs_nearest.append(avg_std_nearest)
    else:
        # Use precomputed statistics
        single_stats = precomputed_stats["single"]
        local_stats = precomputed_stats["local"]
        all_multiframe_stats = precomputed_stats["multiframe"]

        # Extract standard deviations from precomputed stats
        std_devs_blended = []
        std_devs_nearest = []
        for stats in all_multiframe_stats:
            avg_std_blended = np.mean(stats["std_blended"]) * 1000.0
            avg_std_nearest = np.mean(stats["std_nearest"]) * 1000.0
            std_devs_blended.append(avg_std_blended)
            std_devs_nearest.append(avg_std_nearest)

    # Baseline standard deviations (convert to mm/s)
    single_std = np.mean(single_stats["std"]) * 1000.0
    local_std = np.mean(local_stats["std"]) * 1000.0

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        K_range,
        std_devs_blended,
        "o-",
        linewidth=2,
        markersize=8,
        label="Blended",
        color="blue",
    )
    ax.plot(
        K_range,
        std_devs_nearest,
        "s-",
        linewidth=2,
        markersize=8,
        label="Nearest",
        color="purple",
    )
    ax.axhline(
        local_std,
        linestyle="--",
        color="green",
        linewidth=2,
        label="Local Frames",
    )

    ax.set_xlabel("Number of Sampled Frames", fontsize=15)
    ax.set_ylabel("Average Standard Deviation (mm/s)", fontsize=15)
    ax.set_title(
        f'Standard Deviation: Blended vs Nearest Frame ({single_stats["num_experiments"]} experiments)',
        fontsize=15,
    )
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.tick_params(axis="both", which="major", labelsize=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved standard deviation plot to {save_path}")

    plt.show()  # Show the plot to the user

    return fig


def plot_variance_vs_K_multiple_temperatures(
    cached_raw_data: dict,
    K_range: List[int],
    temperatures: List[float],
    save_path: str = None,
) -> plt.Figure:
    """
    Plot variance as a function of K for multiple temperature values.

    Args:
        cached_raw_data: Raw data cache from compute_raw_multiframe_data
        K_range: Range of K values to test
        temperatures: List of temperature values to compare
        save_path: Path to save plot

    Returns:
        Figure object
    """
    # Get baseline statistics
    single_stats = cached_raw_data["single"]
    local_stats = cached_raw_data["local"]

    # Baseline standard deviations (convert to mm/s)
    single_std = np.mean(single_stats["std"]) * 1000.0
    local_std = np.mean(local_stats["std"]) * 1000.0

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color map for temperatures
    colors_temp = ["blue", "purple", "red"]

    # Plot each temperature
    for temp_idx, temp in enumerate(temperatures):
        print(f"\n  Computing variance for temperature={temp}...")
        std_devs_blended = []
        std_devs_nearest = []

        for raw_data_dict in cached_raw_data["multiframe_raw"]:
            K = raw_data_dict["K"]
            if K not in K_range:
                continue

            # Apply temperature blending
            stats = compute_stats_from_raw_data(
                raw_data_dict["raw_data_all"],
                cached_raw_data["transitions_list"],
                temperature=temp,
            )

            # Average standard deviation across all components and time (convert to mm/s)
            avg_std_blended = np.mean(stats["std_blended"]) * 1000.0
            avg_std_nearest = np.mean(stats["std_nearest"]) * 1000.0
            std_devs_blended.append(avg_std_blended)
            std_devs_nearest.append(avg_std_nearest)

        # Plot blended
        ax.plot(
            K_range,
            std_devs_blended,
            "o-",
            linewidth=2,
            markersize=8,
            label=f"Blended (T={temp})",
            color=colors_temp[temp_idx],
        )

        # Plot nearest (dashed)
        ax.plot(
            K_range,
            std_devs_nearest,
            "s--",
            linewidth=2,
            markersize=6,
            label=f"Nearest (T={temp})",
            color=colors_temp[temp_idx],
            alpha=0.6,
        )

    # Plot baselines
    ax.axhline(
        local_std,
        linestyle="--",
        color="green",
        linewidth=2,
        label="Local Frames (K=∞)",
    )

    ax.set_xlabel("Number of Sampled Frames (K)", fontsize=15)
    ax.set_ylabel("Average Standard Deviation (mm/s)", fontsize=15)
    ax.set_title(
        f'Standard Deviation vs K: Temperature Comparison ({single_stats["num_experiments"]} experiments)',
        fontsize=15,
    )
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.tick_params(axis="both", which="major", labelsize=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Saved multi-temperature variance plot to {save_path}")

    return fig


def print_summary_statistics(results: list, K_values: List[int]):
    """Print summary statistics for different K values."""
    transitions_list, valid_indices = get_ground_truth_transitions(results)

    if not transitions_list:
        print("Error: No ground truth transitions found")
        return

    single_stats = compute_velocity_stats_single_frame(
        results, transitions_list, valid_indices
    )
    local_stats = compute_velocity_stats_local(results, transitions_list, valid_indices)

    print(f"\n{'='*70}")
    print("VELOCITY STATISTICS: DISCRETE FRAME APPROXIMATION")
    print(f"{'='*70}")

    # Single frame
    print(f"\nSingle Body-Fixed (K=1):")
    for i, axis in enumerate(["X", "Y", "Z"]):
        mean_val = np.mean(single_stats["mean"][:, i])
        std_val = np.mean(single_stats["std"][:, i])
        print(f"  {axis}: mean={mean_val:.6f}, avg_std={std_val:.6f}")

    # Multi-frame for each K
    for K in K_values:
        stats = compute_velocity_stats_multiframe(
            results, transitions_list, valid_indices, K
        )
        print(f"\nSampled Frames (K={K}) - Blended:")
        for i, axis in enumerate(["X", "Y", "Z"]):
            mean_val = np.mean(stats["mean_blended"][:, i])
            std_val = np.mean(stats["std_blended"][:, i])
            print(f"  {axis}: mean={mean_val:.6f}, avg_std={std_val:.6f}")

        print(f"\nSampled Frames (K={K}) - Nearest:")
        for i, axis in enumerate(["X", "Y", "Z"]):
            mean_val = np.mean(stats["mean_nearest"][:, i])
            std_val = np.mean(stats["std_nearest"][:, i])
            print(f"  {axis}: mean={mean_val:.6f}, avg_std={std_val:.6f}")

    # Local frames
    print(f"\nContinuous Local Frames (K=∞):")
    for i, axis in enumerate(["X", "Y", "Z"]):
        mean_val = np.mean(local_stats["mean"][:, i])
        std_val = np.mean(local_stats["std"][:, i])
        print(f"  {axis}: mean={mean_val:.6f}, avg_std={std_val:.6f}")

    print(f"{'='*70}")


def visualize_sampled_frames(
    results: list,
    K_values: List[int] = [2, 5, 10],
    num_experiments: int = 5,
):
    """
    Visualize sampled reference frames on pointclouds using polyscope.

    Args:
        results: Experiment results
        K_values: List of K values to visualize
        num_experiments: Number of experiments to show
    """
    import polyscope as ps

    ps.init()

    x_offset = 0.15  # Offset between experiments
    z_offset = 0.15  # Offset between K values

    # Get reference pointcloud for computing frame indices
    ref_pointcloud_vertices = results[0]["pointcloud"]["vertices"]

    for k_idx, K in enumerate(K_values):
        # Compute frame indices ONCE from reference pointcloud using FPS
        frame_indices = farthest_point_sampling(ref_pointcloud_vertices, K)

        for exp_idx in range(min(num_experiments, len(results))):
            result = results[exp_idx]
            trajectory = np.array(result["trajectory"])
            pointcloud_vertices = result["pointcloud"]["vertices"]
            pointcloud_local_bases = result["pointcloud"]["local_bases"]

            # Apply offsets (only for experiment and K, not for method)
            offset = np.array([exp_idx * x_offset, 0, k_idx * z_offset])
            pcloud_offset = pointcloud_vertices + offset
            traj_offset = trajectory + offset

            # Register pointcloud and trajectory once per (K, exp) pair
            name_pcloud = f"K{K}_exp{exp_idx}_pcloud"
            ps_pcloud = ps.register_point_cloud(
                name_pcloud,
                pcloud_offset,
                radius=0.005,
                color=(0.8, 0.8, 0.8),
                enabled=True,
            )

            # Add all pointcloud local bases as vector quantities (smaller than sampled frames)
            pcloud_axis_length = 0.008
            # X-axis vectors (red) for all pointcloud points
            ps_pcloud.add_vector_quantity(
                "all_x_axis",
                pointcloud_local_bases[:, :, 0] * pcloud_axis_length,
                enabled=True,
                length=0.03,
                radius=0.005,
                color=(1.0, 0.3, 0.3),
            )
            # Y-axis vectors (green) for all pointcloud points
            ps_pcloud.add_vector_quantity(
                "all_y_axis",
                pointcloud_local_bases[:, :, 1] * pcloud_axis_length,
                enabled=True,
                length=0.03,
                radius=0.005,
                color=(0.3, 1.0, 0.3),
            )
            # Z-axis vectors (blue) for all pointcloud points
            ps_pcloud.add_vector_quantity(
                "all_z_axis",
                pointcloud_local_bases[:, :, 2] * pcloud_axis_length,
                enabled=True,
                length=0.03,
                radius=0.005,
                color=(0.3, 0.3, 1.0),
            )

            # Register trajectory
            name_traj = f"K{K}_exp{exp_idx}_traj"
            ps_curve = ps.register_curve_network(
                name_traj,
                traj_offset,
                edges=np.array([[i, i + 1] for i in range(len(traj_offset) - 1)]),
                radius=0.002,
                color=(0.3, 0.3, 1.0),
                enabled=True,
            )

            # Extract frames at the SAME indices for this experiment's pointcloud
            frame_positions, frame_orientations = sample_frames_from_pointcloud(
                pointcloud_vertices, pointcloud_local_bases, frame_indices
            )

            frames_offset = frame_positions + offset

            # Register sampled frame positions
            name_frames = f"K{K}_exp{exp_idx}_frames"
            ps_frames = ps.register_point_cloud(
                name_frames,
                frames_offset,
                radius=0.01,
                color=(1.0, 0.5, 0.0),  # Orange
                enabled=True,
            )

            # Add frame axes as vector quantities
            axis_length = 0.015
            # X-axis vectors (red)
            ps_frames.add_vector_quantity(
                "x_axis",
                frame_orientations[:, :, 0] * axis_length * 2,
                enabled=True,
                length=0.07,
                radius=0.012,
                color=(1.0, 0.0, 0.0),
            )
            # Y-axis vectors (green)
            ps_frames.add_vector_quantity(
                "y_axis",
                frame_orientations[:, :, 1] * axis_length * 2,
                enabled=True,
                length=0.07,
                radius=0.012,
                color=(0.0, 1.0, 0.0),
            )
            # Z-axis vectors (blue)
            ps_frames.add_vector_quantity(
                "z_axis",
                frame_orientations[:, :, 2] * axis_length * 2,
                enabled=True,
                length=0.07,
                radius=0.012,
                color=(0.0, 0.0, 1.0),
            )

            # Compute and visualize blended frames at all trajectory points
            # Compute frame weights for all trajectory points
            weights = compute_frame_weights(
                trajectory, frame_positions, temperature=0.05
            )

            # Find nearest sampled frame for each trajectory point
            nearest_frame_indices = np.zeros(len(trajectory), dtype=int)
            nearest_orientations = np.zeros((len(trajectory), 3, 3))
            for i in range(len(trajectory)):
                distances = np.linalg.norm(frame_positions - trajectory[i], axis=1)
                nearest_idx = np.argmin(distances)
                nearest_frame_indices[i] = nearest_idx
                nearest_orientations[i] = frame_orientations[nearest_idx]

            # Blend frame orientations at each trajectory point
            blended_orientations = np.zeros((len(trajectory), 3, 3))
            for i in range(len(trajectory)):
                # Weighted combination of frame orientations
                for k in range(K):
                    blended_orientations[i] += weights[i, k] * frame_orientations[k]

                # Orthonormalize the blended orientation using Gram-Schmidt
                # This ensures the axes remain orthogonal and unit length
                for col in range(3):
                    # Subtract projections onto previous axes
                    for prev_col in range(col):
                        projection = np.dot(
                            blended_orientations[i, :, col],
                            blended_orientations[i, :, prev_col],
                        )
                        blended_orientations[i, :, col] -= (
                            projection * blended_orientations[i, :, prev_col]
                        )
                    # Normalize
                    norm = np.linalg.norm(blended_orientations[i, :, col])
                    if norm > 1e-8:
                        blended_orientations[i, :, col] /= norm

            # Apply offset
            traj_blended_offset = trajectory + offset

            # Visualize blended frames
            plot_orientation_field(
                traj_blended_offset,
                blended_orientations,
                name=f"K{K}_exp{exp_idx}_blended",
                vector_length=0.012,
                vector_radius=0.008,
                point_radius=0.006,
                enable_vector=True,
                enable_x=True,
            )

            # Visualize nearest frame orientations at trajectory points
            plot_orientation_field(
                traj_blended_offset,
                nearest_orientations,
                name=f"K{K}_exp{exp_idx}_nearest",
                vector_length=0.010,
                vector_radius=0.006,
                point_radius=0.004,
                enable_vector=False,
                enable_x=True,
            )

    print(f"\n✓ Visualizing sampled frames:")
    print(f"  K values: {K_values}")
    print(f"  Experiments: {min(num_experiments, len(results))}")
    print(f"  Gray points = pointcloud")
    print(f"  Light RGB axes (small) = all pointcloud local bases")
    print(f"  Blue curve = trajectory")
    print(f"  Orange spheres = sampled frame positions (FPS)")
    print(f"  Bright RGB axes (large) = sampled frame orientations at FPS positions")
    print(f"  Trajectory points with blended frames (medium RGB axes)")
    print(f"  Trajectory points with nearest frames (small RGB x-axis only)")
    ps.show()


def main(temperature: float = 0.005):
    """
    Main execution.

    Args:
        temperature: Temperature value for softmax blending (default: 0.001)
    """
    import pickle
    from pathlib import Path

    # Load results
    results = load_results("peeling_batch_results.pkl")

    # Get plots directory
    plots_dir = get_plots_dir()

    # K values to test
    K_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # K_range for variance plot: powers of two
    K_range = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    # Print statistics
    # print_summary_statistics(results, K_values)

    print(f"\n{'='*80}")
    print(f"TEMPERATURE PARAMETER: {temperature}")
    print(f"{'='*80}\n")

    # Cache file path (temperature-independent)
    cache_file = Path("peeling_sampled_frames_raw_data_cache.pkl")

    # Try to load cached raw data
    cached_stats = None
    if cache_file.exists():
        print(f"\n✓ Found cached statistics at {cache_file}")
        try:
            with open(cache_file, "rb") as f:
                cached_stats = pickle.load(f)
            # Verify cache has correct K values
            if cached_stats["K_values"] == K_range:
                print("  ✓ Cache is valid, using precomputed statistics")
            else:
                print("  ✗ Cache has different K values, recomputing...")
                cached_stats = None
        except Exception as e:
            print(f"  ✗ Error loading cache: {e}, recomputing...")
            cached_stats = None

    # Compute statistics if not cached
    if cached_stats is None:
        print("\n✓ Computing statistics for all K values...")
        transitions_list, valid_indices = get_ground_truth_transitions(results)

        if not transitions_list:
            print("Error: No ground truth transitions found")
            return

        # Compute baseline statistics once
        single_stats = compute_velocity_stats_single_frame(
            results, transitions_list, valid_indices
        )
        local_stats = compute_velocity_stats_local(
            results, transitions_list, valid_indices
        )

        # Compute multiframe statistics for all K values once
        # OPTIMIZATION: Reuse frame indices from previous K values
        # Sample up to max(K_range) once, then reuse subsets
        print(f"  Sampling frames up to K={max(K_range)} using FPS...")
        ref_idx = valid_indices[0]
        ref_result = results[ref_idx]
        ref_pointcloud_vertices = ref_result["pointcloud"]["vertices"]
        source_vertices = ref_result["source_vertices"]

        # Sample maximum K frames ONCE
        max_K = max(K_range)
        all_frame_indices = farthest_point_sampling(
            ref_pointcloud_vertices, max_K, seed_indices=source_vertices
        )

        # Cache raw data (temperature-independent)
        multiframe_raw_data_list = []
        for K in K_range:
            print(f"  Computing raw data for K={K}...")
            # Reuse first K frames from the precomputed sample
            frame_indices_K = all_frame_indices[:K]

            # Compute raw data for all experiments
            raw_data_all = []

            for idx in valid_indices:
                result = results[idx]
                trajectory = np.array(result["trajectory"])
                pointcloud_vertices = result["pointcloud"]["vertices"]
                pointcloud_local_bases = result["pointcloud"]["local_bases"]

                # Extract frames at the SAME pointcloud indices
                frame_positions = pointcloud_vertices[frame_indices_K]
                frame_orientations = pointcloud_local_bases[frame_indices_K]

                # Compute temperature-independent raw data
                raw_data = compute_raw_multiframe_data(
                    trajectory, frame_positions, frame_orientations
                )
                raw_data_all.append(raw_data)

            # Store raw data
            multiframe_raw_data_list.append(
                {
                    "K": K,
                    "frame_indices": frame_indices_K,
                    "raw_data_all": raw_data_all,
                }
            )

        # Create cached data dictionary (raw data only)
        cached_raw_data = {
            "single": single_stats,
            "local": local_stats,
            "multiframe_raw": multiframe_raw_data_list,
            "K_values": K_range,
            "transitions_list": transitions_list,
            "valid_indices": valid_indices,
        }

        # Save to cache
        print(f"\n✓ Saving raw data to cache: {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(cached_raw_data, f)
        print(f"  Cache size: {cache_file.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        # Use cached raw data
        cached_raw_data = cached_stats

    # Now compute statistics for the current temperature from cached raw data
    print(f"\n✓ Computing statistics with temperature={temperature}...")
    multiframe_stats_list = []
    for raw_data_dict in cached_raw_data["multiframe_raw"]:
        K = raw_data_dict["K"]
        print(f"  Applying temperature blending for K={K}...")
        stats = compute_stats_from_raw_data(
            raw_data_dict["raw_data_all"],
            cached_raw_data["transitions_list"],
            temperature=temperature,
        )
        stats["num_frames"] = K
        stats["frame_indices"] = raw_data_dict["frame_indices"]
        multiframe_stats_list.append(stats)

    # Construct the full cached_stats structure for compatibility
    cached_stats = {
        "single": cached_raw_data["single"],
        "local": cached_raw_data["local"],
        "multiframe": multiframe_stats_list,
        "K_values": cached_raw_data["K_values"],
    }

    # Create visualizations
    print("\n✓ Generating plots...")

    # Comparison for selected K values (blended)
    plot_comparison_multiple_K(
        results,
        K_values=K_values,
        save_path=plots_dir / "peeling_velocity_sampled_frames_comparison_blended.pdf",
        precomputed_stats=cached_stats,
    )

    # Comparison for selected K values (nearest frame)
    plot_comparison_nearest_frame(
        results,
        K_values=K_values,
        save_path=plots_dir / "peeling_velocity_sampled_frames_comparison_nearest.pdf",
        precomputed_stats=cached_stats,
    )

    # Variance vs K curve
    plot_variance_vs_K(
        results,
        K_range=K_range,
        save_path=plots_dir / f"peeling_variance_vs_K_T{temperature}.pdf",
        precomputed_stats=cached_stats,
    )

    # Save variance data for later combination
    print(f"\n✓ Saving variance data for temperature={temperature}...")
    variance_data = {
        "temperature": temperature,
        "K_range": K_range,
        "std_devs_blended": [
            np.mean(stats["std_blended"]) * 1000.0
            for stats in cached_stats["multiframe"]
        ],
        "std_devs_nearest": [
            np.mean(stats["std_nearest"]) * 1000.0
            for stats in cached_stats["multiframe"]
        ],
        "single_std": np.mean(cached_stats["single"]["std"]) * 1000.0,
        "local_std": np.mean(cached_stats["local"]["std"]) * 1000.0,
        "num_experiments": cached_stats["single"]["num_experiments"],
    }

    variance_file = plots_dir / f"variance_data_T{temperature}.pkl"
    with open(variance_file, "wb") as f:
        pickle.dump(variance_data, f)
    print(f"  Saved variance data to {variance_file}")

    # Visualize sampled frames in 3D using FPS
    # print("\n✓ Launching polyscope visualization...")
    # visualize_sampled_frames(
    #     results,
    #     K_values=[50],
    #     num_experiments=1,
    # )

    print("\n✓ Analysis complete!")


def combine_variance_plots(temperature_values: List[float], save_path: str = None):
    """
    Combine variance data from multiple temperature runs into a single plot.

    Args:
        temperature_values: List of temperature values to combine
        save_path: Path to save the combined plot
    """
    import pickle
    from pathlib import Path

    plots_dir = get_plots_dir()

    # Load variance data for each temperature
    variance_data_list = []
    for temp in temperature_values:
        variance_file = plots_dir / f"variance_data_T{temp}.pkl"
        if not variance_file.exists():
            print(f"Warning: Variance data for T={temp} not found at {variance_file}")
            print(f"  Please run: main(temperature={temp}) first")
            continue

        with open(variance_file, "rb") as f:
            variance_data_list.append(pickle.load(f))

    if not variance_data_list:
        print("Error: No variance data files found!")
        return None

    # Get baseline stats from first file (they should all be the same)
    local_std = variance_data_list[0]["local_std"]
    num_experiments = variance_data_list[0]["num_experiments"]
    K_range = variance_data_list[0]["K_range"]

    # Create plot (narrower width)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color map for temperatures
    colors_temp = ["blue", "purple", "red", "orange", "cyan", "magenta"]

    # Plot nearest frame line once (it's the same for all temperatures)
    ax.plot(
        K_range,
        variance_data_list[0]["std_devs_nearest"],
        "s--",
        linewidth=2,
        markersize=6,
        label="Nearest",
        color="red",
        alpha=0.7,
    )

    # Plot blended for each temperature (excluding T=0.005)
    color_idx = 0
    for variance_data in variance_data_list:
        temp = variance_data["temperature"]

        # Skip T=0.005
        if temp == 0.005:
            continue

        color = colors_temp[color_idx % len(colors_temp)]
        color_idx += 1

        # Plot blended
        ax.plot(
            K_range,
            variance_data["std_devs_blended"],
            "o-",
            linewidth=2,
            markersize=8,
            label=f"Blended (T={temp})",
            color=color,
        )

    # Plot baseline
    ax.axhline(
        local_std,
        linestyle="--",
        color="green",
        linewidth=2,
        label="Local Frames",
    )

    ax.set_xlabel("Number of Frames", fontsize=18)
    ax.set_ylabel("Avg. Std. Dev. (mm/s)", fontsize=18)
    ax.set_title(
        f"Standard Deviation: Temperature Comparison ({num_experiments} experiments)",
        fontsize=18,
    )
    ax.legend(fontsize=16, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.tick_params(axis="both", which="major", labelsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Saved combined temperature plot to {save_path}")

    return fig


if __name__ == "__main__":
    main()
