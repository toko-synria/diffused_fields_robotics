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
original_keypoint_position = pcloud.vertices[controller.source_vertices]
# -----------------------------
# Run Experiments
# -----------------------------
for i in range(num_diffusions):
    diffusion_scalar = diffusion_scalar_arr[i]
    # Initialize controller to identify keypoints once
    for seed in range(num_noises):
        np.random.seed(seed)
        # Sample noise for keypoints
        noise = np.random.normal(scale=0.02, size=(2, 3))
        noisy_keypoint_position = original_keypoint_position + noise

        # Find closest points to noisy keypoints to ensure it is on the pointcloud
        _, source_vertices = pcloud.get_closest_points(noisy_keypoint_position)
        keypoint_position = pcloud.vertices[source_vertices]

        controller = Slicing(
            pcloud, source_vertices=source_vertices, diffusion_scalar=diffusion_scalar
        )
        controller.run()
        # controller.visualize_trajectory()  # Commented out for batch processing
        # ps.show()

        all_data.append(
            {
                "seed": seed,
                "noisy_keypoint_position": keypoint_position,
                "diffusion_scalar": diffusion_scalar,
                "trajectory": controller.trajectory,
            }
        )

    print(f"Completed diffusion scalar {i+1}/{num_diffusions}: {diffusion_scalar:.2f}")


# -----------------------------
# Save to File
# -----------------------------
filepath = get_batch_results_path("slicing_diffusion_scalar_keypoints.pkl")
with open(filepath, "wb") as f:
    pickle.dump(all_data, f)

print(f"\nSaved {len(all_data)} results to {filepath}")
