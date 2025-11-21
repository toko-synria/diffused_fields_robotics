"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Shared utility functions for batch experiment analysis.

This module provides common functions used across different batch analysis scripts
for loading results, aligning sequences, and extracting ground truth transitions.
"""

import pickle
from pathlib import Path
import numpy as np
from typing import List, Tuple

from ..core.config import get_batch_results_path


def load_results(filename: str = "peeling_batch_results.pkl") -> list:
    """
    Load batch experiment results from pickle file.

    Args:
        filename: Name of the pickle file containing batch results

    Returns:
        List of experiment result dictionaries
    """
    filepath = get_batch_results_path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    with open(filepath, "rb") as f:
        results = pickle.load(f)

    print(f"✓ Loaded {len(results)} experiments from {filepath}")
    return results


def get_ground_truth_transitions(results: list) -> Tuple[List[np.ndarray], List[int]]:
    """
    Extract ground truth transition indices from experiment results.

    Transitions mark period boundaries in cyclical tasks (e.g., peeling cycles).

    Args:
        results: List of experiment result dictionaries

    Returns:
        Tuple of (transitions_list, valid_indices) where:
            - transitions_list: List of transition index arrays for each valid experiment
            - valid_indices: Indices of experiments that have transition data
    """
    transitions_all = []
    valid_indices = []

    for i, result in enumerate(results):
        if 'transition_indices' in result and len(result['transition_indices']) > 0:
            transitions_all.append(np.array(result['transition_indices']))
            valid_indices.append(i)

    print(f"\n{'='*60}")
    print(f"GROUND TRUTH TRANSITION ALIGNMENT")
    print(f"{'='*60}")
    print(f"Found {len(valid_indices)}/{len(results)} experiments with ground truth transitions")
    if transitions_all:
        num_transitions = [len(t) for t in transitions_all]
        print(f"Transitions per experiment: min={min(num_transitions)}, max={max(num_transitions)}, mean={np.mean(num_transitions):.1f}")

    return transitions_all, valid_indices


def pad_segment(segment: np.ndarray, target_len: int) -> np.ndarray:
    """
    Pad a segment with its last value to reach target length.

    Args:
        segment: Data segment to pad (length, features)
        target_len: Target length after padding

    Returns:
        Padded segment of shape (target_len, features)
    """
    if len(segment) == 0:
        return np.zeros((target_len, segment.shape[1] if len(segment.shape) > 1 else 3))

    pad_len = target_len - len(segment)
    if pad_len <= 0:
        return segment[:target_len]

    pad_values = np.tile(segment[-1:], (pad_len, 1))
    return np.vstack([segment, pad_values])


def segment_and_pad(
    data: np.ndarray,
    transitions: np.ndarray,
    ref_transitions: np.ndarray,
    ref_len: int
) -> np.ndarray:
    """
    Segment data at transitions and pad each segment to match reference lengths.

    This enables alignment of sequences with different lengths by:
    1. Splitting both data and reference at transition points
    2. Padding each data segment to match corresponding reference segment length

    Args:
        data: Data array to segment and pad (length, features)
        transitions: Transition indices for this data
        ref_transitions: Reference transition indices
        ref_len: Total length of reference sequence

    Returns:
        Aligned data array of shape (ref_len, features)
    """
    segments = []
    split_points = [0] + list(transitions) + [len(data)]
    ref_split_points = [0] + list(ref_transitions) + [ref_len]

    for i in range(len(ref_split_points) - 1):
        src_start = split_points[i]
        src_end = split_points[i + 1]
        ref_seg_len = ref_split_points[i + 1] - ref_split_points[i]

        seg = data[src_start:src_end]
        seg_padded = pad_segment(seg, ref_seg_len)
        segments.append(seg_padded)

    return np.vstack(segments)


def align_by_transitions(
    velocities_list: List[np.ndarray],
    transitions_list: List[np.ndarray]
) -> Tuple[np.ndarray, int]:
    """
    Align multiple sequences using transition-based segment padding.

    This aligns sequences of different lengths by:
    1. Selecting the longest sequence as reference
    2. Using transition indices to split sequences into periods
    3. Padding each period in other sequences to match reference period lengths

    Args:
        velocities_list: List of velocity arrays (varying lengths, 3)
        transitions_list: List of transition index arrays

    Returns:
        Tuple of (aligned_array, ref_index) where:
            - aligned_array: Aligned data (num_experiments, ref_length, features)
            - ref_index: Index of the reference sequence used for alignment
    """
    # Choose reference (longest trajectory)
    ref_idx = max(range(len(velocities_list)), key=lambda i: len(velocities_list[i]))
    ref_data = velocities_list[ref_idx]
    ref_trans = transitions_list[ref_idx]
    ref_len = len(ref_data)

    aligned_all = []
    for data, trans in zip(velocities_list, transitions_list):
        aligned = segment_and_pad(data, trans, ref_trans, ref_len)
        aligned_all.append(aligned)

    return np.array(aligned_all), ref_idx
