"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

from diffused_fields.manifold import Pointcloud

from diffused_fields_robotics.local_action_primitives.action_primitives import Slicing

filename = "banana_half.ply"
pcloud = Pointcloud(filename=filename)


controller = Slicing(pcloud)
controller.run()
controller.visualize_trajectory(show_tool=True, num_samples=5)
