"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Visualize batch peeling experiment results.

This script provides visualization utilities for batch peeling experiments,
including individual experiment visualization and combined multi-experiment views.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def load_results(filename: str = "peeling_batch_results_new.pkl") -> list:
    """Load batch experiment results."""
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filename}")

    with open(filepath, "rb") as f:
        results = pickle.load(f)

    print(f"✓ Loaded {len(results)} experiments from {filename}")
    return results


def visualize_all_experiments(results: list, experiments_per_row: int = 5):
    """
    Visualize all experiments together in polyscope.

    For more than 5 experiments, arranges them in rows with:
    - X-axis offset within each row
    - Z-axis offset between rows

    Args:
        results: List of all experiment results
        experiments_per_row: Number of experiments per row (default: 5)
    """
    import polyscope as ps

    print("\n✓ Visualizing all experiments in polyscope...")

    # Initialize polyscope
    ps.init()

    # Offset distances
    x_offset_distance = 0.13  # 10 cm offset between experiments in a row
    z_offset_distance = 0.1  # 15 cm offset between rows

    for i, result in enumerate(results):
        if "pointcloud" not in result or "trajectory" not in result:
            print(f"Warning: Experiment {i} missing visualization data")
            continue

        # Calculate row and column for this experiment
        row = i // experiments_per_row
        col = i % experiments_per_row

        # Calculate offset for this experiment
        # X-axis: within row, Z-axis: between rows
        offset = np.array([col * x_offset_distance, 0, row * z_offset_distance])

        # Add pointcloud with offset (black color)
        pcloud_data = result["pointcloud"]
        vertices_offset = pcloud_data["vertices"] + offset
        ps.register_point_cloud(
            f"pcloud_exp{i}",
            vertices_offset,
            color=[0, 0, 0],  # Black color
            enabled=True,
        )

        # Add trajectory with offset (red color, thicker)
        trajectory = result["trajectory"]
        trajectory_offset = trajectory + offset
        ps.register_curve_network(
            f"trajectory_exp{i}",
            trajectory_offset,
            np.array([[j, j + 1] for j in range(len(trajectory) - 1)]),
            color=[1, 0, 0],  # Red color
            radius=0.01,  # 1 cm radius
            enabled=True,
        )

        # Add start/end points with offset
        ps.register_point_cloud(
            f"start_exp{i}",
            trajectory_offset[0:1],
            color=[0, 1, 0],  # Green for start
            radius=0.005,
            enabled=True,
        )
        ps.register_point_cloud(
            f"end_exp{i}",
            trajectory_offset[-1:],
            color=[1, 0, 0],  # Red for end
            radius=0.005,
            enabled=True,
        )

    num_rows = (len(results) + experiments_per_row - 1) // experiments_per_row
    print(f"✓ Loaded {len(results)} experiments")
    print(
        f"  Arranged in {num_rows} row(s) with up to {experiments_per_row} experiments per row"
    )
    print("\nLegend:")
    for i, result in enumerate(results):
        row = i // experiments_per_row
        col = i % experiments_per_row
        scale = result.get("scale_factors", None)
        twist = result.get("twist_strength", None)
        points = result.get("trajectory_length", "N/A")
        scale_str = (
            f"[{scale[0]:.2f},{scale[1]:.2f},{scale[2]:.2f}]"
            if scale is not None
            else "N/A"
        )
        twist_str = f"{twist:.2f}" if twist is not None else "N/A"
        print(
            f"  Exp {i} (Row {row}, Col {col}): {points} pts, Scale:{scale_str}, Twist:{twist_str}"
        )

    print("\n✓ Showing polyscope (close window to exit)...")
    ps.show()


def visualize_individual_experiment(result_index: int, results: list):
    """
    Visualize an individual experiment directly from batch results data.

    Args:
        result_index: Index of the experiment to visualize
        results: List of all experiment results
    """
    if result_index < 0 or result_index >= len(results):
        print(
            f"Error: Invalid experiment index {result_index}. Valid range: 0-{len(results)-1}"
        )
        return

    result = results[result_index]

    # Check if we have all required data for visualization
    required_keys = ["trajectory", "trajectory_bases", "pointcloud"]
    missing_keys = [key for key in required_keys if key not in result]

    if missing_keys:
        print(f"Error: Missing required data for visualization: {missing_keys}")
        print(f"This batch file may be from an older version.")
        return

    # Import visualization tools
    from diffused_fields_robotics.local_action_primitives.action_primitives import (
        pcloudActionPrimitives,
    )

    print(f"\n✓ Loading experiment {result_index} from batch results")

    # Reconstruct results_data format expected by visualize_from_results
    results_data = {
        "primitive_type": result.get("primitive_type", "peeling"),
        "object_name": result.get("object_name", "unknown"),
        "trajectory": result["trajectory"],
        "trajectory_local_bases": result["trajectory_bases"],
        "source_vertices": result.get("source_vertices", []),
        "parameters": result.get("parameters", {}),
        "pointcloud": result["pointcloud"],
        "timestamp": f"Exp {result.get('exp_idx', 'N/A')}, Seed {result.get('seed', 'N/A')}",
    }

    # Display experiment info
    print("\n" + "=" * 60)
    print(f"EXPERIMENT {result_index} DETAILS")
    print("=" * 60)
    print(
        f"Exp Index: {result.get('exp_idx', 'N/A')}, Seed: {result.get('seed', 'N/A')}"
    )
    print(f"Diffusion Scalar: {result.get('diffusion_scalar', 'N/A')}")
    print(f"Trajectory Length: {result.get('trajectory_length', 'N/A')} points")

    scale = result.get("scale_factors", None)
    twist = result.get("twist_strength", None)
    if scale is not None:
        print(f"Scale Factors: [{scale[0]:.2f}, {scale[1]:.2f}, {scale[2]:.2f}]")
    if twist is not None:
        print(f"Twist Strength: {twist:.2f}")
    print("=" * 60)

    # Visualize
    print("\n✓ Starting visualization (close window to continue)...")
    pcloudActionPrimitives.visualize_from_results(
        results_data, show_tool=True, num_samples=4
    )


def list_experiments(results: list):
    """List all experiments with their metadata."""
    print("\n" + "=" * 60)
    print("AVAILABLE EXPERIMENTS")
    print("=" * 60)
    for i, result in enumerate(results):
        exp_idx = result.get("exp_idx", "N/A")
        seed = result.get("seed", "N/A")
        diffusion = result.get("diffusion_scalar", "N/A")
        traj_len = result.get("trajectory_length", "N/A")
        scale_factors = result.get("scale_factors", None)
        twist = result.get("twist_strength", None)

        # Format deformation info
        deform_info = ""
        if scale_factors is not None:
            scale_str = f"[{scale_factors[0]:.2f},{scale_factors[1]:.2f},{scale_factors[2]:.2f}]"
            deform_info = f" Scale:{scale_str}"
        if twist is not None:
            deform_info += f" Twist:{twist:.2f}"

        print(
            f"[{i}] Exp:{exp_idx} Seed:{seed} Diffusion:{diffusion} "
            f"Points:{traj_len}{deform_info}"
        )
    print("=" * 60)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Visualize batch peeling experiment results"
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="peeling_batch_results.pkl",
        # default="slicing_batch_results_new.pkl",
        help="Path to batch results pickle file (default: peeling_batch_results_new.pkl)",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        metavar="INDEX",
        help="Visualize a specific experiment by index (0-based). If not specified, visualizes all experiments.",
    )

    args = parser.parse_args()

    # Load results
    results = load_results(args.filename)

    # Always list experiments first
    list_experiments(results)

    # Visualize specific experiment if requested, otherwise show all
    if args.experiment is not None:
        visualize_individual_experiment(args.experiment, results)
    else:
        visualize_all_experiments(results)


if __name__ == "__main__":
    main()
