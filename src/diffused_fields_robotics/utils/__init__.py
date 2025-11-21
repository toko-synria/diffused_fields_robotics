"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Utilities module for diffused_fields_robotics.

This module provides common utilities, base classes, and helper functions
to reduce code duplication across the package.
"""

from .experiment_base import BaseBatchExperiment, BatchSlicingBase, BatchPeelingBase, BatchCoverageBase
from .noise_generation import (
    generate_keypoint_noise,
    generate_geometric_noise,
    generate_scaling_factors,
    generate_topological_noise
)
from .coordinate_utils import (
    compute_body_fixed_coordinate_system,
    apply_coordinate_transformation
)
from .batch_analysis import (
    load_results,
    get_ground_truth_transitions,
    align_by_transitions,
    pad_segment,
    segment_and_pad
)
from .factory import create_primitive_controller

__all__ = [
    'BaseBatchExperiment',
    'BatchSlicingBase',
    'BatchPeelingBase',
    'BatchCoverageBase',
    'generate_keypoint_noise',
    'generate_geometric_noise',
    'generate_scaling_factors',
    'generate_topological_noise',
    'compute_body_fixed_coordinate_system',
    'apply_coordinate_transformation',
    'load_results',
    'get_ground_truth_transitions',
    'align_by_transitions',
    'pad_segment',
    'segment_and_pad',
    'create_primitive_controller'
]