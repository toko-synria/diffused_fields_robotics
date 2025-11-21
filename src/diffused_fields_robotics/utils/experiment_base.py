"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

"""
Base classes for experiments to eliminate code duplication.
"""

import pickle
import numpy as np
from typing import Callable, Optional, Dict, Any, List

from diffused_fields.manifold import Pointcloud
from ..core.config import get_batch_results_path

# Import action primitives - these exist but may have import issues
# We'll import them dynamically to handle missing dependencies gracefully
Cutting = Slicing = Peeling = Coverage = None

def _import_action_primitives():
    """Dynamically import action primitives with error handling."""
    global Cutting, Slicing, Peeling, Coverage
    try:
        from ..local_action_primitives.action_primitives import (
            Cutting, Slicing, Peeling, Coverage
        )
        return True
    except ImportError as e:
        print(f"Warning: Could not import action primitives: {e}")
        return False

# Try to import action primitives at module level
_import_action_primitives()


class BaseBatchExperiment:
    """
    Base class for batch experiments that eliminates common boilerplate code.
    
    This class handles:
    - Common experiment parameters setup
    - Point cloud initialization
    - Nested experiment loops
    - Data collection and storage
    - Results saving
    """
    
    def __init__(
        self,
        filename: str = "banana_half.ply",
        num_experiments: int = 10,
        num_samples: int = 50,
        diffusion_scalar: float = 1000,
        diffusion_range: tuple = None,
        random_seed: int = 42
    ):
        """
        Initialize base batch experiment.

        Args:
            filename: Point cloud file to load
            num_experiments: Number of experiments to run (varying parameters)
            num_samples: Number of random noise samples per experiment
            diffusion_scalar: Fixed diffusion scalar value (default: 1000)
            diffusion_range: Optional (min, max) for diffusion scalar logspace.
                           If specified, overrides fixed diffusion_scalar
            random_seed: Random seed for reproducibility (default: 42)
        """
        self.filename = filename
        self.num_experiments = num_experiments
        self.num_samples = num_samples
        self.diffusion_range = diffusion_range
        self.random_seed = random_seed

        # Setup experiment parameters
        if diffusion_range is not None:
            # Use varying diffusion scalars if range is specified
            self.diffusion_scalar_arr = np.logspace(
                np.log10(diffusion_range[0]),
                np.log10(diffusion_range[1]),
                num_experiments
            )
            print(f"Using varying diffusion_scalar from {diffusion_range[0]} to {diffusion_range[1]}")
        else:
            # Use fixed diffusion scalar for all experiments (default behavior)
            self.diffusion_scalar_arr = np.full(num_experiments, diffusion_scalar)
            print(f"Using fixed diffusion_scalar: {diffusion_scalar} for all experiments")

        # Initialize data storage
        self.all_data = []

        # Initialize point cloud
        self.pcloud = Pointcloud(filename=filename)

        print(f"Initialized batch experiment with {filename}")
        print(f"Experiments: {num_experiments}, Samples per experiment: {num_samples}")
    
    def run_experiment_loop(
        self,
        experiment_func: Callable,
        save_filename: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[Any, Any]]:
        """
        Run the standard nested experiment loop.

        Args:
            experiment_func: Function to run for each (exp_idx, sample_idx) combination
            save_filename: Optional filename to save results
            progress_callback: Optional callback for progress tracking

        Returns:
            List of experiment results
        """
        total_experiments = self.num_experiments * self.num_samples
        current_exp = 0

        for exp_idx in range(self.num_experiments):
            # Set random seed once at the start of each experiment
            # Each experiment gets a different base seed
            np.random.seed(self.random_seed + exp_idx)

            diffusion_scalar = self.diffusion_scalar_arr[exp_idx]

            for sample_idx in range(self.num_samples):
                # Don't reset seed here - let RNG state advance naturally
                # This gives us different noise samples from the same seeded RNG

                # Run the specific experiment
                result = experiment_func(exp_idx, sample_idx)

                # Add common metadata
                result.update({
                    "exp_idx": exp_idx,
                    "sample_idx": sample_idx,
                    "diffusion_scalar": diffusion_scalar,
                })

                self.all_data.append(result)

                # Progress tracking
                current_exp += 1
                if progress_callback:
                    progress_callback(current_exp, total_experiments)
                else:
                    if current_exp % 10 == 0 or current_exp == total_experiments:
                        print(f"Progress: {current_exp}/{total_experiments} experiments completed")

        # Save results if filename provided
        if save_filename:
            self.save_results(save_filename)

        return self.all_data
    
    def save_results(self, filename: str):
        """Save experiment results to pickle file."""
        filepath = get_batch_results_path(filename)
        print(f"Saving {len(self.all_data)} results to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(self.all_data, f)
        print(f"Results saved successfully")

    def load_results(self, filename: str) -> List[Dict]:
        """Load experiment results from pickle file."""
        filepath = get_batch_results_path(filename)
        with open(filepath, "rb") as f:
            self.all_data = pickle.load(f)
        print(f"Loaded {len(self.all_data)} results from {filepath}")
        return self.all_data
    
    def compute_rmse_analysis(self) -> Dict[str, float]:
        """
        Compute basic RMSE analysis on trajectory data.
        Override this method for specific analysis needs.
        """
        if not self.all_data:
            return {}
            
        rmse_values = []
        for result in self.all_data:
            if 'trajectory' in result:
                # Simple RMSE computation - override for specific needs
                trajectory = np.array(result['trajectory'])
                if len(trajectory) > 1:
                    diffs = np.diff(trajectory, axis=0)
                    rmse = np.sqrt(np.mean(np.sum(diffs**2, axis=1)))
                    rmse_values.append(rmse)
        
        if rmse_values:
            return {
                'mean_rmse': np.mean(rmse_values),
                'std_rmse': np.std(rmse_values),
                'min_rmse': np.min(rmse_values),
                'max_rmse': np.max(rmse_values)
            }
        return {}



class BatchSlicingBase(BaseBatchExperiment):
    """
    Specialized base class for slicing experiments with common slicing setup.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Try to get reference keypoints from a slicing controller
        if Slicing is not None:
            try:
                controller = Slicing(self.pcloud)
                self.original_keypoints = self.pcloud.vertices[controller.source_vertices]
                self.source_vertices = controller.source_vertices
                print(f"✓ Slicing base initialized with {len(self.source_vertices)} keypoints")
            except Exception as e:
                print(f"Warning: Could not initialize Slicing controller: {e}")
                self._use_fallback_keypoints(kwargs)
        else:
            self._use_fallback_keypoints(kwargs)
    
    def _use_fallback_keypoints(self, kwargs):
        """Use fallback keypoints when Slicing is not available."""
        self.source_vertices = kwargs.get('source_vertices', [0, len(self.pcloud.vertices)//2])
        self.original_keypoints = self.pcloud.vertices[self.source_vertices]
        print(f"✓ Using fallback keypoints: {len(self.source_vertices)} vertices")


class BatchPeelingBase(BaseBatchExperiment):
    """
    Specialized base class for peeling experiments.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Try to get reference keypoints from a peeling controller
        if Peeling is not None:
            try:
                controller = Peeling(self.pcloud)
                self.source_vertices = controller.source_vertices
                print(f"✓ Peeling base initialized with {len(self.source_vertices)} keypoints")
            except Exception as e:
                print(f"Warning: Could not initialize Peeling controller: {e}")
                self._use_fallback_keypoints(kwargs)
        else:
            self._use_fallback_keypoints(kwargs)
    
    def _use_fallback_keypoints(self, kwargs):
        """Use fallback keypoints when Peeling is not available."""
        self.source_vertices = kwargs.get('source_vertices', [0, len(self.pcloud.vertices)//4])
        print(f"✓ Using fallback keypoints: {len(self.source_vertices)} vertices")


class BatchCoverageBase(BaseBatchExperiment):
    """
    Specialized base class for coverage experiments.
    Coverage typically uses boundary vertices as sources.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Try to get reference keypoints from a coverage controller
        if Coverage is not None:
            try:
                controller = Coverage(self.pcloud)
                # Coverage uses boundary vertices, which may be empty initially
                self.source_vertices = controller.source_vertices if hasattr(controller, 'source_vertices') else []
                print(f"✓ Coverage base initialized with {len(self.source_vertices)} boundary keypoints")
            except Exception as e:
                print(f"Warning: Could not initialize Coverage controller: {e}")
                self._use_fallback_keypoints(kwargs)
        else:
            self._use_fallback_keypoints(kwargs)

    def _use_fallback_keypoints(self, kwargs):
        """Use fallback keypoints when Coverage is not available."""
        # Coverage doesn't require source vertices - it uses boundaries
        self.source_vertices = kwargs.get('source_vertices', [])
        print(f"✓ Using fallback keypoints: {len(self.source_vertices)} vertices (empty for boundary-based coverage)")