"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

import pickle

import numpy as np

from diffused_fields.manifold import Pointcloud
from diffused_fields_robotics.local_action_primitives.action_primitives import Slicing
from diffused_fields_robotics.core.config import get_batch_results_path

# -----------------------------
# Experiment Setup
# -----------------------------
filename = "banana_half.ply"

num_noises = 50
num_diffusions = 10
diffusion_scalar_arr = np.logspace(np.log10(0.1), np.log10(10000), num_diffusions)

all_data = []

# Get fresh copy of the point cloud
pcloud = Pointcloud(filename=filename)
controller = Slicing(pcloud)
source_vertices = controller.source_vertices

# -----------------------------
# Run Experiments
# -----------------------------
for i in range(num_diffusions):
    diffusion_scalar = diffusion_scalar_arr[i]
    for seed in range(num_noises):
        np.random.seed(seed)
        # Get fresh copy of the point cloud
        pcloud_noisy = Pointcloud(filename=filename)

        # Add Gaussian noise
        noise_std = 0.003  # standard deviation of the noise
        noise = np.random.normal(scale=noise_std, size=pcloud_noisy.vertices.shape)
        pcloud_noisy.vertices = pcloud_noisy.vertices + noise

        controller = Slicing(pcloud_noisy, source_vertices=source_vertices, diffusion_scalar=diffusion_scalar)
        controller.run()
        # controller.visualize_trajectory()  # Commented out for batch processing
        # ps.show()

        all_data.append(
            {
                "seed": seed,
                "noise": noise,
                "diffusion_scalar": diffusion_scalar,
                "trajectory": controller.trajectory,
            }
        )

    print(f"Completed diffusion scalar {i+1}/{num_diffusions}: {diffusion_scalar:.2f}")

# -----------------------------
# Save to File
# -----------------------------
filepath = get_batch_results_path("slicing_diffusion_scalar_geometric_noise.pkl")
with open(filepath, "wb") as f:
    pickle.dump(all_data, f)

print(f"\nSaved {len(all_data)} results to {filepath}")
