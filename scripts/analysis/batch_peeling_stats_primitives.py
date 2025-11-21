"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Batch peeling statistics using primitive-aligned cylindrical coordinates.

This script analyzes velocities in a cylindrical coordinate system aligned with
the peeling primitive, where:
- Longitudinal axis (along primitive): x-axis of body-fixed frame
- Radial direction: distance from longitudinal axis
- Tangential direction: perpendicular to both

Compares three representations:
1. Cartesian body-fixed: Standard body-fixed frame (x, y, z)
2. Cylindrical primitive: (longitudinal, radial, tangential)
3. Local frames: Continuous position-varying frames
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from diffused_fields_robotics.core.config import get_plots_dir
from diffused_fields_robotics.utils import (
    align_by_transitions,
    get_ground_truth_transitions,
    load_results,
)


def compute_cylindrical_velocities(
    velocities_body_fixed: np.ndarray,
    trajectory: np.ndarray,
    body_fixed_frame_R: np.ndarray,
    body_fixed_frame_origin: np.ndarray = None,
) -> np.ndarray:
    """
    Convert body-fixed Cartesian velocities to cylindrical coordinates.

    Cylindrical coordinate system aligned with x-axis (longitudinal direction):
    - Longitudinal (x): Along x-axis of body-fixed frame (primitive direction)
    - Tangential (theta): Circumferential around x-axis
    - Radial (r): Away from x-axis in the y-z plane

    Args:
        velocities_body_fixed: Velocities in body-fixed Cartesian frame (N, 3)
        trajectory: Trajectory points in global frame (N+1, 3)
        body_fixed_frame_R: Body-fixed rotation matrix (3, 3)
        body_fixed_frame_origin: Origin of body-fixed frame in global coords (3,)
                                  If None, uses first trajectory point

    Returns:
        velocities_cylindrical: (N, 3) array [v_x, v_theta, v_r]
    """
    N = len(velocities_body_fixed)
    velocities_cylindrical = np.zeros((N, 3))

    # Use first trajectory point as origin if not specified
    if body_fixed_frame_origin is None:
        body_fixed_frame_origin = trajectory[0]

    # Transform trajectory to body-fixed frame to get positions
    x_bf = body_fixed_frame_R[:, 0]
    y_bf = body_fixed_frame_R[:, 1]
    z_bf = body_fixed_frame_R[:, 2]

    for i in range(N):
        v_bf = velocities_body_fixed[i]  # Velocity in body-fixed Cartesian (vx, vy, vz)
        pos_global = trajectory[i]

        # Position in body-fixed frame (translate then rotate)
        pos_relative = pos_global - body_fixed_frame_origin
        pos_bf = np.array(
            [
                np.dot(pos_relative, x_bf),
                np.dot(pos_relative, y_bf),
                np.dot(pos_relative, z_bf),
            ]
        )

        # Longitudinal velocity (x-component in body-fixed frame)
        v_long = v_bf[0]

        # Radial direction: position in y-z plane of body-fixed frame
        r_yz = np.array([pos_bf[1], pos_bf[2]])  # (y, z) position
        r_norm = np.linalg.norm(r_yz)

        if r_norm > 1e-6:
            # Unit radial direction in y-z plane
            r_hat_yz = r_yz / r_norm  # (cos(θ), sin(θ))

            # Radial velocity: projection of (vy, vz) onto radial direction
            v_rad = v_bf[1] * r_hat_yz[0] + v_bf[2] * r_hat_yz[1]

            # Tangential direction: perpendicular to radial in y-z plane
            # t_hat = (-sin(θ), cos(θ)) = perpendicular to r_hat
            t_hat_yz = np.array([-r_hat_yz[1], r_hat_yz[0]])

            # Tangential velocity: projection of (vy, vz) onto tangential direction
            v_tang = v_bf[1] * t_hat_yz[0] + v_bf[2] * t_hat_yz[1]
        else:
            # On the longitudinal axis: radial/tangential are ambiguous
            # Assign transverse velocity magnitude to radial, zero to tangential
            v_rad = np.sqrt(v_bf[1] ** 2 + v_bf[2] ** 2)
            v_tang = 0.0

        # Return in order [x, theta, r] with reversed radial direction
        velocities_cylindrical[i] = [v_long, v_tang, v_rad]

    return velocities_cylindrical


def compute_spherical_velocities(
    velocities_body_fixed: np.ndarray,
    trajectory: np.ndarray,
    body_fixed_frame_R: np.ndarray,
    body_fixed_frame_origin: np.ndarray = None,
) -> np.ndarray:
    """
    Convert body-fixed Cartesian velocities to spherical coordinates.

    Spherical coordinate system centered at origin of body-fixed frame:
    - Elevation (theta): Polar angle from x-axis (0 to π)
    - Azimuth (phi): Azimuthal angle in y-z plane from y-axis (0 to 2π)
    - Radial (r): Distance from origin

    Velocity components:
    - v_theta: Elevation velocity (perpendicular to r in x-r plane)
    - v_phi: Azimuthal velocity (perpendicular to both, around x-axis)
    - v_r: Radial velocity (along position vector)

    Args:
        velocities_body_fixed: Velocities in body-fixed Cartesian frame (N, 3)
        trajectory: Trajectory points in global frame (N+1, 3)
        body_fixed_frame_R: Body-fixed rotation matrix (3, 3)
        body_fixed_frame_origin: Origin of body-fixed frame in global coords (3,)
                                  If None, uses first trajectory point

    Returns:
        velocities_spherical: (N, 3) array [v_theta, v_phi, v_r]
    """
    N = len(velocities_body_fixed)
    velocities_spherical = np.zeros((N, 3))

    # Use first trajectory point as origin if not specified
    if body_fixed_frame_origin is None:
        body_fixed_frame_origin = trajectory[0]

    # Body-fixed frame axes
    x_bf = body_fixed_frame_R[:, 0]
    y_bf = body_fixed_frame_R[:, 1]
    z_bf = body_fixed_frame_R[:, 2]

    for i in range(N):
        v_bf = velocities_body_fixed[i]  # Velocity in body-fixed Cartesian (vx, vy, vz)
        pos_global = trajectory[i]

        # Position in body-fixed frame (translate then rotate)
        pos_relative = pos_global - body_fixed_frame_origin
        pos_bf = np.array(
            [
                np.dot(pos_relative, x_bf),
                np.dot(pos_relative, y_bf),
                np.dot(pos_relative, z_bf),
            ]
        )

        # Spherical coordinates from Cartesian position
        r = np.linalg.norm(pos_bf)

        if r > 1e-6:
            # Radial direction: unit vector along position
            r_hat = pos_bf / r

            # Radial velocity: projection of velocity onto radial direction
            v_r = np.dot(v_bf, r_hat)

            # Transverse component: velocity perpendicular to radial
            v_trans = v_bf - v_r * r_hat

            # Compute distance from x-axis (projection onto y-z plane)
            r_yz = np.sqrt(pos_bf[1] ** 2 + pos_bf[2] ** 2)

            if r_yz > 1e-6:
                # Theta direction: perpendicular to r, pointing away from x-axis
                # In spherical coords with theta from x-axis:
                # theta_hat = (∂r/∂θ)/|∂r/∂θ| where r = (r sin θ cos φ, r sin θ sin φ, r cos θ)
                # After normalization: theta_hat = (cos θ cos φ, cos θ sin φ, -sin θ)
                # In our frame: cos θ = x/r, sin θ = r_yz/r
                cos_theta = pos_bf[0] / r
                sin_theta = r_yz / r

                # Unit vector in y-z plane
                cos_phi = pos_bf[1] / r_yz
                sin_phi = pos_bf[2] / r_yz

                theta_hat = np.array(
                    [
                        -sin_theta,  # -sin(θ)
                        cos_theta * cos_phi,  # cos(θ) * cos(φ)
                        cos_theta * sin_phi,  # cos(θ) * sin(φ)
                    ]
                )

                # Polar velocity
                v_theta = np.dot(v_bf, theta_hat)

                # Azimuthal direction: perpendicular to both r and theta
                # phi_hat = (-sin φ, cos φ, 0) in the y-z plane
                phi_hat = np.array([0.0, -sin_phi, cos_phi])  # -sin(φ)  # cos(φ)

                # Azimuthal velocity
                v_phi = np.dot(v_bf, phi_hat)
            else:
                # On x-axis: theta is 0 or π, phi is ambiguous
                # Assign transverse velocity magnitude to theta component
                v_theta = np.sqrt(v_bf[1] ** 2 + v_bf[2] ** 2)
                v_phi = 0.0
        else:
            # At origin: all directions are ambiguous
            v_r = np.linalg.norm(v_bf)
            v_theta = 0.0
            v_phi = 0.0

        # Return in order [elevation, azimuth, r] with reversed azimuth and radial directions
        velocities_spherical[i] = [-v_theta, v_phi, v_r]

    return velocities_spherical


def compute_velocity_stats(
    results: list,
    transitions_list: List[np.ndarray],
    valid_indices: List[int],
    velocity_type: str,
) -> dict:
    """
    Unified function to compute velocity statistics for different coordinate systems.

    Args:
        results: List of experiment results
        transitions_list: Ground truth transition indices
        valid_indices: Valid experiment indices
        velocity_type: 'cartesian', 'cylindrical', 'spherical', or 'local'

    Returns:
        Dictionary with velocity statistics
    """
    velocities_all = []

    for idx in valid_indices:
        result = results[idx]

        if velocity_type == "cartesian":
            if "velocities_body_fixed" in result:
                velocities_all.append(result["velocities_body_fixed"])
        elif velocity_type == "cylindrical":
            velocities_body_fixed = result["velocities_body_fixed"]
            trajectory = np.array(result["trajectory"])
            body_fixed_frame_R = result["body_fixed_frame_R"]

            # Compute point cloud center as origin
            if "pointcloud" in result and "vertices" in result["pointcloud"]:
                pcloud_vertices = result["pointcloud"]["vertices"]
                origin = np.mean(pcloud_vertices, axis=0)
            else:
                origin = None  # Will default to trajectory[0]

            vel = compute_cylindrical_velocities(
                velocities_body_fixed, trajectory, body_fixed_frame_R, origin
            )
            velocities_all.append(vel)
        elif velocity_type == "spherical":
            velocities_body_fixed = result["velocities_body_fixed"]
            trajectory = np.array(result["trajectory"])
            body_fixed_frame_R = result["body_fixed_frame_R"]

            # Compute point cloud center as origin
            if "pointcloud" in result and "vertices" in result["pointcloud"]:
                pcloud_vertices = result["pointcloud"]["vertices"]
                origin = np.mean(pcloud_vertices, axis=0)
            else:
                origin = None  # Will default to trajectory[0]

            vel = compute_spherical_velocities(
                velocities_body_fixed, trajectory, body_fixed_frame_R, origin
            )
            velocities_all.append(vel)
        elif velocity_type == "local":
            if "velocities_local" in result:
                velocities_all.append(result["velocities_local"])
        else:
            raise ValueError(f"Unknown velocity type: {velocity_type}")

    if not velocities_all:
        return None

    velocities_aligned, ref_idx = align_by_transitions(velocities_all, transitions_list)

    # Friendly type names for printing
    type_names = {
        "cartesian": "Cartesian Body-Fixed",
        "cylindrical": "Cylindrical Primitive",
        "spherical": "Spherical Primitive",
        "local": "Local Frames",
    }
    print(
        f"{type_names[velocity_type]}: Aligned {len(velocities_all)} demos using reference {ref_idx}"
    )

    return {
        "velocities": velocities_aligned,
        "mean": velocities_aligned.mean(axis=0),
        "std": velocities_aligned.std(axis=0),
        "max_len": velocities_aligned.shape[1],
        "num_experiments": len(velocities_aligned),
        "transitions": transitions_list,
        "ref_index": ref_idx,
        "valid_indices": valid_indices,
    }


def extract_periodic_cycles(
    velocities_list: List[np.ndarray],
    transitions_list: List[np.ndarray],
    num_cycles: int = 3,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract individual periodic cycles from full trajectories.

    Each trajectory has num_cycles repetitions. We split at cycle boundaries
    and return individual cycles with their sub-transitions.

    Args:
        velocities_list: List of velocity arrays, one per experiment
        transitions_list: List of transition indices, one per experiment
        num_cycles: Number of periodic cycles in each trajectory (default: 3)

    Returns:
        cycles_list: List of individual cycle velocity arrays
        cycle_transitions_list: List of transition indices for each cycle
    """
    cycles_list = []
    cycle_transitions_list = []

    for velocities, transitions in zip(velocities_list, transitions_list):
        # Determine transitions per cycle
        total_transitions = len(transitions)
        transitions_per_cycle = total_transitions // num_cycles

        for cycle_idx in range(num_cycles):
            # Get transition indices for this cycle
            start_trans_idx = cycle_idx * transitions_per_cycle
            end_trans_idx = (cycle_idx + 1) * transitions_per_cycle

            # Get data boundaries for this cycle
            # First cycle starts at 0, subsequent cycles start after previous cycle's last transition
            if cycle_idx == 0:
                cycle_start = 0
            else:
                cycle_start = transitions[start_trans_idx - 1]

            # Last cycle ends at trajectory end, others end at their last transition
            if cycle_idx == num_cycles - 1:
                cycle_end = len(velocities)
            else:
                cycle_end = transitions[end_trans_idx - 1]

            # Extract cycle data
            cycle_velocities = velocities[cycle_start:cycle_end]

            # Extract and re-index transitions for this cycle (relative to cycle start)
            cycle_transitions = transitions[start_trans_idx:end_trans_idx] - cycle_start

            cycles_list.append(cycle_velocities)
            cycle_transitions_list.append(cycle_transitions)

    return cycles_list, cycle_transitions_list


def compute_velocity_stats_periodic(
    results: list,
    transitions_list: List[np.ndarray],
    valid_indices: List[int],
    velocity_type: str,
    num_cycles: int = 3,
) -> dict:
    """
    Compute velocity statistics for periodic cycles.

    Extracts individual cycles from all experiments and aligns them,
    resulting in num_experiments * num_cycles aligned cycles.

    Args:
        results: List of experiment results
        transitions_list: Ground truth transition indices
        valid_indices: Valid experiment indices
        velocity_type: 'cartesian', 'cylindrical', 'spherical', or 'local'
        num_cycles: Number of cycles per trajectory

    Returns:
        Dictionary with velocity statistics for periodic cycles
    """
    # Get velocities based on type
    velocities_all = []

    for idx in valid_indices:
        result = results[idx]

        if velocity_type == "cartesian":
            velocities_all.append(result["velocities_body_fixed"])
        elif velocity_type == "cylindrical":
            velocities_body_fixed = result["velocities_body_fixed"]
            trajectory = np.array(result["trajectory"])
            body_fixed_frame_R = result["body_fixed_frame_R"]

            # Compute point cloud center as origin
            if "pointcloud" in result and "vertices" in result["pointcloud"]:
                pcloud_vertices = result["pointcloud"]["vertices"]
                origin = np.mean(pcloud_vertices, axis=0)
            else:
                origin = None  # Will default to trajectory[0]

            vel_cyl = compute_cylindrical_velocities(
                velocities_body_fixed, trajectory, body_fixed_frame_R, origin
            )
            velocities_all.append(vel_cyl)
        elif velocity_type == "spherical":
            velocities_body_fixed = result["velocities_body_fixed"]
            trajectory = np.array(result["trajectory"])
            body_fixed_frame_R = result["body_fixed_frame_R"]

            # Compute point cloud center as origin
            if "pointcloud" in result and "vertices" in result["pointcloud"]:
                pcloud_vertices = result["pointcloud"]["vertices"]
                origin = np.mean(pcloud_vertices, axis=0)
            else:
                origin = None  # Will default to trajectory[0]

            vel_sph = compute_spherical_velocities(
                velocities_body_fixed, trajectory, body_fixed_frame_R, origin
            )
            velocities_all.append(vel_sph)
        elif velocity_type == "local":
            velocities_all.append(result["velocities_local"])
        else:
            raise ValueError(f"Unknown velocity type: {velocity_type}")

    # Extract periodic cycles
    cycles_list, cycle_transitions_list = extract_periodic_cycles(
        velocities_all, transitions_list, num_cycles
    )

    # Align all cycles
    aligned_cycles, ref_idx = align_by_transitions(cycles_list, cycle_transitions_list)

    num_total_cycles = len(cycles_list)
    print(
        f"{velocity_type.capitalize()} Periodic ({num_cycles} cycles): "
        f"Aligned {num_total_cycles} cycles from {len(valid_indices)} experiments"
    )

    return {
        "velocities": aligned_cycles,
        "mean": aligned_cycles.mean(axis=0),
        "std": aligned_cycles.std(axis=0),
        "max_len": aligned_cycles.shape[1],
        "num_experiments": len(valid_indices),
        "num_cycles": num_total_cycles,
        "transitions": cycle_transitions_list,
        "ref_index": ref_idx,
        "valid_indices": valid_indices,
    }


def plot_four_way_comparison(results: list, save_path: str = None) -> plt.Figure:
    """Plot velocity comparison: Cartesian BF vs Cylindrical vs Spherical vs Local frames."""
    # Set font to Arial
    plt.rcParams["font.family"] = "Arial"

    # Get ground truth transitions
    transitions_list, valid_indices = get_ground_truth_transitions(results)

    if not transitions_list:
        print("Error: No ground truth transitions found")
        return None

    # Compute statistics for all four approaches
    cartesian_stats = compute_velocity_stats(
        results, transitions_list, valid_indices, "cartesian"
    )
    cylindrical_stats = compute_velocity_stats(
        results, transitions_list, valid_indices, "cylindrical"
    )
    spherical_stats = compute_velocity_stats(
        results, transitions_list, valid_indices, "spherical"
    )
    local_stats = compute_velocity_stats(
        results, transitions_list, valid_indices, "local"
    )

    if (
        cartesian_stats is None
        or cylindrical_stats is None
        or spherical_stats is None
        or local_stats is None
    ):
        print("Error: Could not compute velocity statistics")
        return None

    # Setup plot - 4 columns x 3 rows
    fig, axes = plt.subplots(3, 4, figsize=(20, 8), sharex=True)

    # Column labels and colors
    colors = ["red", "green", "blue"]

    # Axis labels for different coordinate systems
    axis_labels_list = [
        ["X", "Y", "Z"],  # Cartesian
        ["x", "θ", "r"],  # Cylindrical (longitudinal, tangential, radial)
        ["θ", "φ", "r"],  # Spherical (elevation, azimuth, radial)
        ["X", "Y", "Z"],  # Local frames
    ]

    stats_list = [cartesian_stats, cylindrical_stats, spherical_stats, local_stats]
    titles = [
        "Cartesian",
        "Cylindrical",
        "Spherical",
        "Local",
    ]

    # Convert to mm/s (multiply by 1000)
    scale_factor = 1000.0

    for col, (stats, title, axis_labels) in enumerate(
        zip(stats_list, titles, axis_labels_list)
    ):
        for i in range(3):  # Component index
            ax = axes[i, col]

            ax.plot(
                stats["mean"][:, i] * scale_factor,
                color=colors[i],
                linewidth=2,
                label="mean",
            )

            ax.fill_between(
                np.arange(stats["max_len"]),
                (stats["mean"][:, i] - stats["std"][:, i]) * scale_factor,
                (stats["mean"][:, i] + stats["std"][:, i]) * scale_factor,
                color=colors[i],
                alpha=0.3,
                label="±1 std",
            )

            ax.axhline(0, linestyle="--", color="gray", linewidth=1)
            ax.set_ylim(-1.2, 1.2)
            ax.grid(True, alpha=0.3)

            # Y-axis label only on leftmost column
            if col == 0:
                ax.set_ylabel(f"{axis_labels[i]} (mm/s)", fontsize=16)
            else:
                ax.set_yticklabels([])

            # Set tick label size
            ax.tick_params(axis="both", which="major", labelsize=16)

            # Mark period boundaries with vertical dashed lines
            ref_idx = stats["ref_index"]
            transitions = stats["transitions"][ref_idx]
            valid_transitions = transitions[transitions < len(stats["mean"])]

            # Show transitions 5, 10, 15 (indices 4, 9, 14) - end of each of 3 periods
            # Each peeling period has 5 transitions, so show indices 4, 9, 14
            period_boundary_indices = [4, 9, 14]
            period_transitions = [
                valid_transitions[i]
                for i in period_boundary_indices
                if i < len(valid_transitions)
            ]

            # Also add a line at the end of the trajectory
            period_transitions.append(len(stats["mean"]) - 1)

            for j, trans_idx in enumerate(period_transitions):
                label = "Period" if (j == 0 and i == 0) else ""
                ax.axvline(
                    trans_idx,
                    color="purple",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label=label,
                )

            # Title and legend on top row
            if i == 0:
                ax.set_title(title, fontsize=16)
                # Legend only in first column
                if col == 0 and len(period_transitions) > 0:
                    ax.legend(loc="upper right", fontsize=10)

    # X-axis labels on bottom row
    for col in range(4):
        axes[2, col].set_xlabel("Time Step", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved velocity plot to {save_path}")

    return fig


def plot_periodic_cycle_comparison(
    results: list, save_path: str = None, num_cycles: int = 3
) -> plt.Figure:
    """Plot velocity comparison for aligned periodic cycles."""
    # Get ground truth transitions
    transitions_list, valid_indices = get_ground_truth_transitions(results)

    if not transitions_list:
        print("Error: No ground truth transitions found")
        return None

    # Compute periodic cycle statistics for all four approaches
    cartesian_periodic = compute_velocity_stats_periodic(
        results, transitions_list, valid_indices, "cartesian", num_cycles
    )
    cylindrical_periodic = compute_velocity_stats_periodic(
        results, transitions_list, valid_indices, "cylindrical", num_cycles
    )
    spherical_periodic = compute_velocity_stats_periodic(
        results, transitions_list, valid_indices, "spherical", num_cycles
    )
    local_periodic = compute_velocity_stats_periodic(
        results, transitions_list, valid_indices, "local", num_cycles
    )

    if (
        cartesian_periodic is None
        or cylindrical_periodic is None
        or spherical_periodic is None
        or local_periodic is None
    ):
        print("Error: Could not compute periodic statistics")
        return None

    # Setup plot - 4 columns x 3 rows
    fig, axes = plt.subplots(3, 4, figsize=(20, 8), sharex=True)

    # Column labels and colors
    colors = ["red", "green", "blue"]

    # Axis labels for different coordinate systems
    axis_labels_list = [
        ["X", "Y", "Z"],  # Cartesian
        ["x", "θ", "r"],  # Cylindrical (longitudinal, tangential, radial)
        ["θ", "φ", "r"],  # Spherical (elevation, azimuth, radial)
        ["X", "Y", "Z"],  # Local frames
    ]

    stats_list = [
        cartesian_periodic,
        cylindrical_periodic,
        spherical_periodic,
        local_periodic,
    ]
    titles = [
        "Cartesian",
        "Cylindrical",
        "Spherical",
        "Local",
    ]

    # Convert to mm/s (multiply by 1000)
    scale_factor = 1000.0

    for col, (stats, title, axis_labels) in enumerate(
        zip(stats_list, titles, axis_labels_list)
    ):
        for i in range(3):  # Component index
            ax = axes[i, col]

            # Plot mean ± std
            ax.plot(
                stats["mean"][:, i] * scale_factor,
                color=colors[i],
                linewidth=2,
                label="mean",
            )

            ax.fill_between(
                np.arange(stats["max_len"]),
                (stats["mean"][:, i] - stats["std"][:, i]) * scale_factor,
                (stats["mean"][:, i] + stats["std"][:, i]) * scale_factor,
                color=colors[i],
                alpha=0.3,
                label="±1 std",
            )

            ax.axhline(0, linestyle="--", color="gray", linewidth=1)
            ax.set_ylim(-1.2, 1.2)
            ax.grid(True, alpha=0.3)

            # Y-axis label only on leftmost column
            if col == 0:
                ax.set_ylabel(f"{axis_labels[i]} (mm/s)", fontsize=16)
            else:
                ax.set_yticklabels([])

            # Set tick label size
            ax.tick_params(axis="both", which="major", labelsize=16)

            # Mark transitions with vertical lines across all rows
            ref_idx = stats["ref_index"]
            transitions = stats["transitions"][ref_idx]
            valid_transitions = transitions[transitions < len(stats["mean"])]
            for j, trans_idx in enumerate(valid_transitions):
                label = "Transitions" if (j == 0 and i == 0) else ""
                ax.axvline(
                    trans_idx,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label=label,
                )

            # Title on top row
            if i == 0:
                ax.set_title(title, fontsize=16)
                # Legend only in first column
                if col == 0 and len(valid_transitions) > 0:
                    ax.legend(loc="upper right", fontsize=10)

    # X-axis labels on bottom row
    for col in range(4):
        axes[2, col].set_xlabel("Time Step (within cycle)", fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved periodic cycle plot to {save_path}")

    return fig


def print_summary_statistics(results: list):
    """Print summary statistics comparing all three approaches."""
    transitions_list, valid_indices = get_ground_truth_transitions(results)

    if not transitions_list:
        print("Error: No ground truth transitions found")
        return

    cartesian_stats = compute_velocity_stats(
        results, transitions_list, valid_indices, "cartesian"
    )
    cylindrical_stats = compute_velocity_stats(
        results, transitions_list, valid_indices, "cylindrical"
    )
    spherical_stats = compute_velocity_stats(
        results, transitions_list, valid_indices, "spherical"
    )
    local_stats = compute_velocity_stats(
        results, transitions_list, valid_indices, "local"
    )

    if cartesian_stats and cylindrical_stats and spherical_stats and local_stats:
        print(f"\n{'='*70}")
        print("VELOCITY STATISTICS COMPARISON (Full Trajectories)")
        print(f"{'='*70}")

        approaches = [
            ("Cartesian Body-Fixed (x,y,z)", cartesian_stats, ["X", "Y", "Z"]),
            (
                "Cylindrical Primitive (long,rad,tang)",
                cylindrical_stats,
                ["Long", "Rad", "Tang"],
            ),
            (
                "Spherical Primitive (r,theta,phi)",
                spherical_stats,
                ["Radial", "Polar", "Azimuth"],
            ),
            ("Local Frames (x,y,z)", local_stats, ["X", "Y", "Z"]),
        ]

        for approach_name, stats, axis_names in approaches:
            print(f"\n{approach_name}:")
            for i, axis in enumerate(axis_names):
                mean_val = np.mean(stats["mean"][:, i])
                std_val = np.mean(stats["std"][:, i])
                print(f"  {axis}: mean={mean_val:.6f}, avg_std={std_val:.6f}")

        print(f"{'='*70}")

    # Print periodic cycle statistics
    print(f"\n{'='*70}")
    print("PERIODIC CYCLE STATISTICS (Aligned Cycles)")
    print(f"{'='*70}")

    cartesian_periodic = compute_velocity_stats_periodic(
        results, transitions_list, valid_indices, "cartesian"
    )
    cylindrical_periodic = compute_velocity_stats_periodic(
        results, transitions_list, valid_indices, "cylindrical"
    )
    spherical_periodic = compute_velocity_stats_periodic(
        results, transitions_list, valid_indices, "spherical"
    )
    local_periodic = compute_velocity_stats_periodic(
        results, transitions_list, valid_indices, "local"
    )

    if (
        cartesian_periodic
        and cylindrical_periodic
        and spherical_periodic
        and local_periodic
    ):
        approaches_periodic = [
            ("Cartesian Body-Fixed (x,y,z)", cartesian_periodic, ["X", "Y", "Z"]),
            (
                "Cylindrical Primitive (long,rad,tang)",
                cylindrical_periodic,
                ["Long", "Rad", "Tang"],
            ),
            (
                "Spherical Primitive (r,theta,phi)",
                spherical_periodic,
                ["Radial", "Polar", "Azimuth"],
            ),
            ("Local Frames (x,y,z)", local_periodic, ["X", "Y", "Z"]),
        ]

        for approach_name, stats, axis_names in approaches_periodic:
            print(f"\n{approach_name}:")
            for i, axis in enumerate(axis_names):
                mean_val = np.mean(stats["mean"][:, i])
                std_val = np.mean(stats["std"][:, i])
                print(f"  {axis}: mean={mean_val:.6f}, avg_std={std_val:.6f}")

        print(f"{'='*70}")

    # Print variance comparison
    print_variance_comparison(
        cartesian_stats,
        cylindrical_stats,
        spherical_stats,
        local_stats,
        cartesian_periodic,
        cylindrical_periodic,
        spherical_periodic,
        local_periodic,
    )


def print_variance_comparison(
    cart_full,
    cyl_full,
    sph_full,
    local_full,
    cart_periodic,
    cyl_periodic,
    sph_periodic,
    local_periodic,
):
    """Print comparison of average standard deviations and percentage improvements."""
    print(f"\n{'='*70}")
    print("VARIANCE COMPARISON & IMPROVEMENT ANALYSIS")
    print(f"{'='*70}")

    # Full trajectory comparison
    print("\n1. FULL TRAJECTORY STATISTICS:")
    print("-" * 70)

    # Compute average std across all components
    approaches_full = [
        ("Cartesian Body-Fixed", cart_full),
        ("Cylindrical Primitive", cyl_full),
        ("Spherical Primitive", sph_full),
        ("Local Frames", local_full),
    ]

    avg_stds_full = []
    for name, stats in approaches_full:
        # Average std across all components and time
        avg_std = np.mean(stats["std"])
        avg_stds_full.append((name, avg_std))
        print(f"{name:30s}: avg_std = {avg_std:.6f}")

    # Compute percentage improvement over baseline (Cartesian)
    baseline_std_full = avg_stds_full[0][1]
    print(f"\n{'Improvement over Cartesian Body-Fixed:':30s}")
    for i, (name, avg_std) in enumerate(avg_stds_full[1:], 1):
        improvement = (baseline_std_full - avg_std) / baseline_std_full * 100
        print(f"  {name:28s}: {improvement:+6.2f}%")

    # Periodic cycle comparison
    print(f"\n2. PERIODIC CYCLE STATISTICS:")
    print("-" * 70)

    approaches_periodic = [
        ("Cartesian Body-Fixed", cart_periodic),
        ("Cylindrical Primitive", cyl_periodic),
        ("Spherical Primitive", sph_periodic),
        ("Local Frames", local_periodic),
    ]

    avg_stds_periodic = []
    for name, stats in approaches_periodic:
        # Average std across all components and time
        avg_std = np.mean(stats["std"])
        avg_stds_periodic.append((name, avg_std))
        print(f"{name:30s}: avg_std = {avg_std:.6f}")

    # Compute percentage improvement over baseline (Cartesian)
    baseline_std_periodic = avg_stds_periodic[0][1]
    print(f"\n{'Improvement over Cartesian Body-Fixed:':30s}")
    for i, (name, avg_std) in enumerate(avg_stds_periodic[1:], 1):
        improvement = (baseline_std_periodic - avg_std) / baseline_std_periodic * 100
        print(f"  {name:28s}: {improvement:+6.2f}%")

    # Overall summary
    print(f"\n3. BEST PERFORMING APPROACH:")
    print("-" * 70)

    # Find best for full trajectory
    best_full_name, best_full_std = min(avg_stds_full, key=lambda x: x[1])
    best_full_improvement = (
        (baseline_std_full - best_full_std) / baseline_std_full * 100
    )

    # Find best for periodic
    best_periodic_name, best_periodic_std = min(avg_stds_periodic, key=lambda x: x[1])
    best_periodic_improvement = (
        (baseline_std_periodic - best_periodic_std) / baseline_std_periodic * 100
    )

    print(
        f"Full Trajectory  : {best_full_name:28s} ({best_full_improvement:+6.2f}% improvement)"
    )
    print(
        f"Periodic Cycles  : {best_periodic_name:28s} ({best_periodic_improvement:+6.2f}% improvement)"
    )

    print(f"{'='*70}\n")


def main():
    """Main execution."""
    # Load results
    results = load_results("peeling_batch_results.pkl")

    # Print statistics
    print_summary_statistics(results)

    # Get plots directory
    plots_dir = get_plots_dir()

    # Create visualizations
    print("\n✓ Generating velocity plots...")

    # Four-way comparison with mean±std (full trajectories)
    plot_four_way_comparison(
        results,
        save_path=plots_dir / "peeling_velocity_primitives_comparison.pdf",
    )

    # Periodic cycle analysis (150 cycles = 50 experiments × 3 cycles)
    plot_periodic_cycle_comparison(
        results,
        save_path=plots_dir / "peeling_velocity_primitives_periodic.pdf",
        num_cycles=3,
    )

    print("\n✓ Analysis complete! Close plot windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()
