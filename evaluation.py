import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from scipy.spatial.distance import pdist, squareform


class ReconstructionLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, reconstructed, target):
        """
        Args:
            reconstructed: (batch_size, num_features) reconstructed mesh vectors
            target: (batch_size, num_features) original mesh vectors
        
        Returns:
            loss: Scalar loss value
        """
        loss = F.mse_loss(reconstructed, target, reduction=self.reduction)
        return loss


# ============================================================================
# NUMPY-BASED EVALUATION METRICS (for analysis)
# ============================================================================

def compute_reconstruction_error(original: np.ndarray, 
                                reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Compute various reconstruction error metrics
    
    Args:
        original: (N, D) array of original mesh vectors
        reconstructed: (N, D) array of reconstructed mesh vectors
    
    Returns:
        Dictionary of error metrics
    """
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    rmse = np.sqrt(mse)
    
    # Per-sample errors
    per_sample_mse = np.mean((original - reconstructed) ** 2, axis=1)
    per_sample_mae = np.mean(np.abs(original - reconstructed), axis=1)
    
    # Relative error
    original_norm = np.linalg.norm(original, axis=1, keepdims=True)
    relative_error = np.linalg.norm(original - reconstructed, axis=1) / (original_norm.squeeze() + 1e-8)
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'mean_per_sample_mse': float(per_sample_mse.mean()),
        'std_per_sample_mse': float(per_sample_mse.std()),
        'mean_per_sample_mae': float(per_sample_mae.mean()),
        'std_per_sample_mae': float(per_sample_mae.std()),
        'mean_relative_error': float(relative_error.mean()),
        'std_relative_error': float(relative_error.std()),
        'max_error': float(np.max(np.abs(original - reconstructed))),
        'per_sample_errors': per_sample_mse.tolist()
    }


def compute_latent_space_metrics(latent_vectors: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics on latent space representations
    
    Args:
        latent_vectors: (N, latent_dim) array of encoded representations
    
    Returns:
        Dictionary of latent space metrics
    """

    # Compute pairwise distances
    distances = pdist(latent_vectors, metric='euclidean')
    
    # Compute statistics
    mean_distance = distances.mean()
    std_distance = distances.std()
    min_distance = distances.min()
    max_distance = distances.max()
    
    # Compute variance per dimension
    per_dim_variance = latent_vectors.var(axis=0)
    
    # Effective dimensionality (number of dimensions capturing 95% variance)
    sorted_variance = np.sort(per_dim_variance)[::-1]
    cumsum_variance = np.cumsum(sorted_variance)
    total_variance = cumsum_variance[-1]
    effective_dim = np.searchsorted(cumsum_variance, 0.95 * total_variance) + 1
    
    return {
        'mean_pairwise_distance': float(mean_distance),
        'std_pairwise_distance': float(std_distance),
        'min_pairwise_distance': float(min_distance),
        'max_pairwise_distance': float(max_distance),
        'mean_per_dim_variance': float(per_dim_variance.mean()),
        'std_per_dim_variance': float(per_dim_variance.std()),
        'effective_dimensionality': int(effective_dim),
        'total_variance': float(total_variance)
    }


class MeshReconstructionEvaluator:
    """
    Comprehensive evaluator for mesh reconstruction quality
    """
    def __init__(self, num_vertices: int):
        """
        Args:
            num_vertices: Number of vertices per mesh
        """
        self.num_vertices = num_vertices
        self.results = {}
    
    def evaluate(self,
                original: np.ndarray,
                reconstructed: np.ndarray,
                latent_vectors: Optional[np.ndarray] = None) -> Dict:
        """
        Perform comprehensive evaluation
        
        Args:
            original: (N, D) array of original mesh vectors
            reconstructed: (N, D) array of reconstructed mesh vectors
            latent_vectors: Optional (N, latent_dim) array of latent representations
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {}
        
        # Reconstruction error metrics
        results['reconstruction'] = compute_reconstruction_error(original, reconstructed)
        
        # Latent space metrics
        if latent_vectors is not None:
            results['latent_space'] = compute_latent_space_metrics(latent_vectors)
        
        self.results = results
        return results
    
    def print_summary(self):
        """Print a summary of evaluation results"""
        if not self.results:
            print("No evaluation results available. Run evaluate() first.")
            return
        
        print("=" * 70)
        print("MESH RECONSTRUCTION EVALUATION SUMMARY")
        print("=" * 70)
        
        if 'reconstruction' in self.results:
            print("\nReconstruction Error Metrics:")
            r = self.results['reconstruction']
            print(f"  MSE:                  {r['mse']:.6f}")
            print(f"  MAE:                  {r['mae']:.6f}")
            print(f"  RMSE:                 {r['rmse']:.6f}")
            print(f"  Mean Relative Error:  {r['mean_relative_error']:.4f}")
            print(f"  Max Error:            {r['max_error']:.6f}")
        
        if 'latent_space' in self.results:
            print("\nLatent Space Metrics:")
            l = self.results['latent_space']
            print(f"  Mean Pairwise Dist:   {l['mean_pairwise_distance']:.4f}")
            print(f"  Effective Dimensions: {l['effective_dimensionality']}")
            print(f"  Total Variance:       {l['total_variance']:.4f}")
        
        print("=" * 70)
    
    def save_results(self, filepath: str):
        """Save evaluation results to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")


if __name__ == "__main__":
    print("Evaluation and Metrics Module")
    print("="*70)
    
    # Example: Create dummy data
    batch_size = 8
    num_vertices = 1000
    num_features = num_vertices * 3
    latent_dim = 64
    
    # Simulate original and reconstructed data
    original = torch.randn(batch_size, num_features)
    reconstructed = original + torch.randn(batch_size, num_features) * 0.1
    latent = torch.randn(batch_size, latent_dim)
    
    print("\nTesting PyTorch Loss Functions:")
    print("-" * 70)
    
    # Test individual losses
    recon_loss = ReconstructionLoss()
    print(f"Reconstruction Loss: {recon_loss(reconstructed, original).item():.6f}")
    
    # Test numpy-based evaluation
    print("\n" + "="*70)
    print("Testing NumPy Evaluation Metrics:")
    print("-" * 70)
    
    evaluator = MeshReconstructionEvaluator(num_vertices)
    results = evaluator.evaluate(
        original.numpy(),
        reconstructed.numpy(),
        latent.numpy()
    )
    
    evaluator.print_summary()
    
    print("\n" + "="*70)
    print("All metrics ready for use!")
    print("="*70)