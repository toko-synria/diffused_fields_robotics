"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Visualize Force-Torque Data from Peeling Experiments

This script loads and visualizes force-torque measurements collected during
peeling experiments. The data includes 6-axis force-torque sensor readings
(Fx, Fy, Fz, Tx, Ty, Tz) sampled during task execution.

Usage:
    python visualize_ft_data.py [filename]

    If no filename is provided, displays a list of available data files.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diffused_fields_robotics.core.config import get_ft_data_dir, get_plots_dir


def load_ft_data(filepath):
    """
    Load force-torque data from .npy file.

    Args:
        filepath: Path to .npy file containing FT data

    Returns:
        numpy array with shape (N, 6) where N is number of samples
        Columns: [Fx, Fy, Fz, Tx, Ty, Tz]
    """
    data = np.load(filepath)
    print(f"Loaded data from: {filepath}")
    print(f"  Shape: {data.shape}")
    print(f"  Data type: {data.dtype}")

    return data


def plot_ft_data(data, title="Force-Torque Data", save_path=None):
    """
    Create visualization of force-torque data.

    Args:
        data: numpy array with shape (N, 6) or (N, 7)
              If 7 columns, assumes [timestamp, Fx, Fy, Fz, Tx, Ty, Tz]
              If 6 columns, assumes [Fx, Fy, Fz, Tx, Ty, Tz]
        title: Plot title
        save_path: Optional path to save figure
    """
    # Handle different data formats
    if data.shape[1] == 7:
        # Includes timestamp - reset to start from 0
        timestamps = data[:, 0]
        timestamps = timestamps - timestamps[0]  # Start from 0
        forces = data[:, 1:4]
        torques = data[:, 4:7]
        x_label = "Time (s)"
        x_data = timestamps
    elif data.shape[1] == 6:
        # No timestamp, use sample index
        forces = data[:, 0:3]
        torques = data[:, 3:6]
        x_label = "Sample Index"
        x_data = np.arange(len(data))
    else:
        raise ValueError(f"Expected 6 or 7 columns, got {data.shape[1]}")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot forces
    ax1.plot(x_data, forces[:, 0], "r-", label="Fx", alpha=0.7, linewidth=1.5)
    ax1.plot(x_data, forces[:, 1], "g-", label="Fy", alpha=0.7, linewidth=1.5)
    ax1.plot(x_data, forces[:, 2], "b-", label="Fz", alpha=0.7, linewidth=1.5)
    ax1.set_ylabel("Force (N)", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{title} - Forces", fontsize=14, fontweight="bold")

    # Plot torques
    ax2.plot(x_data, torques[:, 0], "r-", label="Tx", alpha=0.7, linewidth=1.5)
    ax2.plot(x_data, torques[:, 1], "g-", label="Ty", alpha=0.7, linewidth=1.5)
    ax2.plot(x_data, torques[:, 2], "b-", label="Tz", alpha=0.7, linewidth=1.5)
    ax2.set_xlabel(x_label, fontsize=12)
    ax2.set_ylabel("Torque (Nm)", fontsize=12)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"{title} - Torques", fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Print statistics
    print(f"\nForce Statistics (N):")
    print(
        f"  Fx: mean={np.mean(forces[:, 0]):.3f}, std={np.std(forces[:, 0]):.3f}, "
        f"max={np.max(np.abs(forces[:, 0])):.3f}"
    )
    print(
        f"  Fy: mean={np.mean(forces[:, 1]):.3f}, std={np.std(forces[:, 1]):.3f}, "
        f"max={np.max(np.abs(forces[:, 1])):.3f}"
    )
    print(
        f"  Fz: mean={np.mean(forces[:, 2]):.3f}, std={np.std(forces[:, 2]):.3f}, "
        f"max={np.max(np.abs(forces[:, 2])):.3f}"
    )

    print(f"\nTorque Statistics (Nm):")
    print(
        f"  Tx: mean={np.mean(torques[:, 0]):.3f}, std={np.std(torques[:, 0]):.3f}, "
        f"max={np.max(np.abs(torques[:, 0])):.3f}"
    )
    print(
        f"  Ty: mean={np.mean(torques[:, 1]):.3f}, std={np.std(torques[:, 1]):.3f}, "
        f"max={np.max(np.abs(torques[:, 1])):.3f}"
    )
    print(
        f"  Tz: mean={np.mean(torques[:, 2]):.3f}, std={np.std(torques[:, 2]):.3f}, "
        f"max={np.max(np.abs(torques[:, 2])):.3f}"
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")

    return fig


def list_available_files():
    """List all available FT data files."""
    ft_data_path = get_ft_data_dir()

    if not ft_data_path.exists():
        print(f"Data directory not found: {ft_data_path}")
        return []

    npy_files = sorted(ft_data_path.glob("*.npy"))

    if not npy_files:
        print(f"No .npy files found in {ft_data_path}")
        return []

    print(f"\nAvailable FT data files in {ft_data_path}:")
    print("-" * 60)
    for i, f in enumerate(npy_files, 1):
        print(f"{i:2d}. {f.name}")
    print("-" * 60)

    return npy_files


def main():
    parser = argparse.ArgumentParser(
        description="Visualize force-torque data from peeling experiments"
    )
    parser.add_argument(
        "filename",
        nargs="?",
        help="Name of .npy file to visualize (e.g., '5_peeling_1.npy')",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save figure to file instead of displaying"
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["png", "pdf"],
        help="Output format for saved plots (default: pdf)",
    )

    args = parser.parse_args()

    # List available files if no filename provided
    if args.filename is None:
        available_files = list_available_files()
        if available_files:
            print("\nUsage: python visualize_ft_data.py <filename>")
            print("Example: python visualize_ft_data.py 5_peeling_1.npy")
        return

    # Construct full path
    ft_data_path = get_ft_data_dir() / args.filename

    if not ft_data_path.exists():
        print(f"Error: File not found: {ft_data_path}")
        print("\nTrying to list available files...")
        list_available_files()
        return

    # Load and visualize data
    data = load_ft_data(ft_data_path)

    # Generate title from filename
    title = args.filename.replace(".npy", "").replace("_", " ").title()

    # Prepare save path if requested
    save_path = None
    if args.save:
        output_dir = get_plots_dir()
        save_path = output_dir / f"{args.filename.replace('.npy', f'.{args.format}')}"

    # Create plot
    fig = plot_ft_data(data, title=title, save_path=save_path)

    # Show plot if not saving
    if not args.save:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
