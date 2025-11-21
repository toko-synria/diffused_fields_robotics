"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
from diffused_fields.manifold import Pointcloud

from diffused_fields_robotics.core.config import get_batch_results_path, get_plots_dir
from diffused_fields_robotics.local_action_primitives.action_primitives import Slicing

# -----------------------------
# Load Point Cloud and Reference Trajectory
# -----------------------------
filename = "banana_half.ply"
pcloud = Pointcloud(filename=filename)
controller = Slicing(pcloud, diffusion_scalar=1000)
controller.run()
reference_trajectory = controller.trajectory


# -----------------------------
# Compute RMSE
# -----------------------------
def compute_rmse(traj1, traj2):
    min_len = min(len(traj1), len(traj2))
    t1 = traj1[:min_len]
    t2 = traj2[:min_len]
    return np.sqrt(np.mean(np.sum((t1 - t2) ** 2, axis=1)))


def load_and_compute_rmse(exp_type):
    """Load data and compute RMSE statistics for a given experiment type."""
    data_filename = f"slicing_diffusion_scalar_{exp_type}.pkl"
    filepath = get_batch_results_path(data_filename)
    with open(filepath, "rb") as f:
        all_data = pickle.load(f)

    # Group by diffusion scalar
    diffusion_groups = {}
    for data in all_data:
        ds = data["diffusion_scalar"]
        trajectory = data["trajectory"]
        rmse = compute_rmse(trajectory, reference_trajectory)
        if ds not in diffusion_groups:
            diffusion_groups[ds] = []
        diffusion_groups[ds].append(rmse)

    # Compute mean and std for each diffusion scalar
    diffusion_scalars = sorted(diffusion_groups.keys())
    rmse_means = [np.mean(diffusion_groups[ds]) for ds in diffusion_scalars]
    rmse_stds = [np.std(diffusion_groups[ds]) for ds in diffusion_scalars]

    return diffusion_scalars, rmse_means, rmse_stds


# -----------------------------
# Load data for all three experiment types
# -----------------------------
exp_types = ["keypoints", "geometric_noise", "topological_noise"]
exp_titles = ["Keypoint Noise", "Geometric Noise", "Topological Noise"]

results = {}
for exp_type in exp_types:
    try:
        results[exp_type] = load_and_compute_rmse(exp_type)
        print(f"Loaded data for {exp_type}")
    except FileNotFoundError:
        print(f"Warning: Data file not found for {exp_type}, skipping...")
        results[exp_type] = None


# -----------------------------
# Create side-by-side plots
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (exp_type, title) in enumerate(zip(exp_types, exp_titles)):
    ax = axes[idx]

    if results[exp_type] is not None:
        diffusion_scalars, rmse_means, rmse_stds = results[exp_type]

        ax.errorbar(
            diffusion_scalars,
            rmse_means,
            yerr=rmse_stds,
            fmt="-o",
            capsize=5,
            color="black",
            label="Mean ± Std",
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\tau$ (log scale)", fontsize=12)
        ax.set_ylabel("RMSE to reference trajectory (m)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            f"No data for\n{title}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlabel(r"$\tau$ (log scale)", fontsize=12)
        ax.set_ylabel("RMSE to reference trajectory (m)", fontsize=12)

plt.tight_layout()
plots_dir = get_plots_dir()
plt.savefig(plots_dir / "robustness_comparison.pdf", dpi=300, bbox_inches="tight")
print(f"\nSaved combined plot to {plots_dir / 'robustness_comparison.pdf'}")
plt.show()


# ps.init()
# if exp_type == "keypoints":
#     keypoint_positions = np.array(keypoint_positions).reshape(-1, 3)
#     plot_orientation_field(keypoint_positions, name="sources")
# elif exp_type == "geometric_noise":
#     pcloud.vertices += noise
#     controller.trajectory = trajectory
# elif exp_type == "topological_noise":
#     import open3d as o3d

#     filename = "banana_half.ply"
#     pcloud = Pointcloud(filename=filename)
#     pcloud.pcd.points = o3d.utility.Vector3dVector(reduced_points)
#     pcloud.vertices = reduced_points
#     pcloud.get_normals()
# controller = Slicing(pcloud, diffusion_scalar=1000)
# controller.run()
# # plot_orientation_field(pcloud.vertices, name="sources")
# controller.visualize_trajectory(show_tool=True, num_samples=0)  # static plot
# ps.show()
