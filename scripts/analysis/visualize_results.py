"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Simple script to visualize saved experiment results without re-running the experiment.

Usage:
    python visualize_results.py <path_to_results_file>

Example:
    python scripts/visualize_results.py results/slicing_clutter_pens_20250924_171510.pkl --show-tool
"""

import argparse
import os
import sys

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from diffused_fields_robotics.local_action_primitives.action_primitives import (
    pcloudActionPrimitives,
)


def main():
    parser = argparse.ArgumentParser(description="Visualize saved experiment results")
    parser.add_argument("results_file", help="Path to the saved results file (.pkl)")
    parser.add_argument(
        "--show-tool",
        action="store_true",
        help="Show tool visualization (if available)",
    )
    parser.add_argument(
        "--num-samples", type=int, help="Number of trajectory samples to show"
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)

    try:
        # Load results
        results_data = pcloudActionPrimitives.load_results(args.results_file)

        # Display experiment info
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Primitive Type: {results_data['primitive_type']}")
        print(f"Object: {results_data['object_name']}")
        print(f"Timestamp: {results_data['timestamp']}")
        print(f"Trajectory Points: {len(results_data['trajectory'])}")

        if "parameters" in results_data:
            print("\nParameters:")
            for key, value in results_data["parameters"].items():
                print(f"  {key}: {value}")

        print("=" * 60)

        # Visualize
        print("\nStarting visualization...")

        # Check if using stored num_samples
        if args.num_samples is None and "parameters" in results_data:
            stored_samples = results_data["parameters"].get(
                "visualization_num_samples", None
            )
            if stored_samples is not None:
                print(f"Using stored visualization samples: {stored_samples}")
            else:
                print("No stored visualization samples found. Using full trajectory.")
        elif args.num_samples is not None:
            print(f"Using specified samples: {args.num_samples}")
        else:
            print("Using full trajectory.")

        pcloudActionPrimitives.visualize_from_results(
            results_data, show_tool=args.show_tool, num_samples=args.num_samples
        )

    except Exception as e:
        print(f"Error loading or visualizing results: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
