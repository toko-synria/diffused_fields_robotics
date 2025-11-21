"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

import pickle

import numpy as np
import open3d as o3d
from diffused_fields.manifold import Pointcloud

from diffused_fields_robotics.core.config import get_batch_results_path
from diffused_fields_robotics.local_action_primitives.action_primitives import Slicing

# -----------------------------
# Experiment Setup
# -----------------------------
filename = "banana_half.ply"

num_noises = 50
num_diffusions = 10
diffusion_scalar_arr = np.logspace(np.log10(0.1), np.log10(10000), num_diffusions)

pcloud = Pointcloud(filename=filename)
controller = Slicing(pcloud, diffusion_scalar=1000)
original_keypoint_positions = pcloud.vertices[controller.source_vertices]
original_start_position = pcloud.vertices[controller.start_vertex]
all_data = []

# -----------------------------
# Run Experiments
# -----------------------------
for i in range(num_diffusions):
    diffusion_scalar = diffusion_scalar_arr[i]
    # Initialize controller to identify keypoints once
    for seed in range(num_noises):
        np.random.seed(seed)
        # Get fresh copy of the point cloud
        pcloud = Pointcloud(filename=filename)
        num_holes = 10  # Number of holes
        hole_radius = 0.005  # Radius of each hole

        points = np.asarray(pcloud.vertices)
        # Select random hole centers from the point cloud
        hole_centers = points[
            np.random.choice(len(points), size=num_holes, replace=False)
        ]

        # Build a mask to keep points outside all holes
        keep_mask = np.ones(len(points), dtype=bool)

        for center in hole_centers:
            distances = np.linalg.norm(points - center, axis=1)
            keep_mask &= distances > hole_radius

        reduced_points = points[keep_mask]
        pcloud.pcd.points = o3d.utility.Vector3dVector(reduced_points)
        pcloud.vertices = reduced_points
        pcloud.get_normals()

        # Find closest points to noisy keypoints to ensure it is on the pointcloud
        _, source_vertices = pcloud.get_closest_points(original_keypoint_positions)
        _, start_vertex = pcloud.get_closest_points(original_start_position)

        controller = Slicing(
            pcloud,
            source_vertices=source_vertices,
            start_vertex=start_vertex,
            diffusion_scalar=diffusion_scalar,
        )
        controller.run()
        # controller.visualize_trajectory()  # Commented out for batch processing
        # ps.show()
        all_data.append(
            {
                "vertices": reduced_points,
                "start_vertex": start_vertex,
                "source_vertices": source_vertices,
                "diffusion_scalar": diffusion_scalar,
                "trajectory": controller.trajectory,
            }
        )

    print(f"Completed diffusion scalar {i+1}/{num_diffusions}: {diffusion_scalar:.2f}")

# -----------------------------
# Save to File
# -----------------------------
filepath = get_batch_results_path("slicing_diffusion_scalar_topological_noise.pkl")
with open(filepath, "wb") as f:
    pickle.dump(all_data, f)

print(f"\nSaved {len(all_data)} results to {filepath}")
