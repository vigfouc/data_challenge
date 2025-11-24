import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from typing import List, Optional
from scipy.spatial.distance import pdist, squareform


def mesh_to_vector(mesh):
    """Convert vertex coordinates into a single (3N,) vector"""
    vertices = np.asarray(mesh.vertices)
    return vertices.flatten()


def vector_to_mesh(vector, reference_mesh):
    """Convert a (3N,) vector back into a mesh"""
    new_mesh = o3d.geometry.TriangleMesh()
    num_vertices = len(vector) // 3
    vertices = vector.reshape((num_vertices, 3))
    new_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    new_mesh.triangles = reference_mesh.triangles
    return new_mesh


def visualize_meshes(meshes: List, window_name: str = "Meshes", colors: Optional[List] = None):
    """
    Visualize multiple meshes with optional colors
    
    Args:
        meshes: List of Open3D meshes
        window_name: Window title
        colors: List of RGB colors (0-1 range) for each mesh
    """
    if colors is not None:
        for mesh, color in zip(meshes, colors):
            mesh.paint_uniform_color(color)
    
    o3d.visualization.draw_geometries(meshes, window_name=window_name)


def visualize_single_mesh(mesh, window_name: str = "Mesh"):
    """Visualize a single mesh"""
    o3d.visualization.draw_geometries([mesh], window_name=window_name)


def visualize_comparison(original_mesh, reconstructed_mesh, 
                        window_name: str = "Original vs Reconstructed"):
    """
    Visualize original (red) vs reconstructed (blue) mesh side by side
    
    Args:
        original_mesh: Original mesh
        reconstructed_mesh: Reconstructed mesh
        window_name: Window title
    """
    original_copy = o3d.geometry.TriangleMesh(original_mesh)
    reconstructed_copy = o3d.geometry.TriangleMesh(reconstructed_mesh)
    
    original_copy.paint_uniform_color([1.0, 0.3, 0.3])  # Red
    reconstructed_copy.paint_uniform_color([0.3, 0.3, 1.0])  # Blue
    
    o3d.visualization.draw_geometries([original_copy, reconstructed_copy],
                                     window_name=window_name)


def visualize_pca(data_matrix: np.ndarray, 
                  filenames: Optional[List[str]] = None,
                  n_components: int = 3, 
                  save_path: Optional[str] = None):
    """
    Visualize PCA decomposition with explained variance and projections
    
    Args:
        data_matrix: (N, D) array of flattened mesh vectors
        filenames: Optional list of filenames for labeling
        n_components: Number of PCA components to compute
        save_path: Optional path to save figure
        
    Returns:
        pca: Fitted PCA object
        transformed: PCA-transformed data
    """
    n_samples = data_matrix.shape[0]
    pca = PCA(n_components=min(n_components, n_samples - 1))
    transformed = pca.fit_transform(data_matrix)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Explained variance
    ax1 = fig.add_subplot(131)
    ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('PCA Explained Variance')
    ax1.grid(True, alpha=0.3)
    
    # 2D projection
    ax2 = fig.add_subplot(132)
    scatter = ax2.scatter(transformed[:, 0], transformed[:, 1], 
                         c=range(n_samples), cmap='viridis', s=100)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax2.set_title('PCA 2D Projection')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Sample Index')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return pca, transformed


def visualize_training_history(train_losses: List[float], 
                               val_losses: Optional[List[float]] = None,
                               metric_name: str = "Loss",
                               save_path: Optional[str] = None):
    """
    Visualize training history
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: Optional list of validation losses per epoch
        metric_name: Name of the metric being plotted
        save_path: Optional path to save figure
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label=f'Training {metric_name}', linewidth=2)
    
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label=f'Validation {metric_name}', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_latent_space(latent_vectors: np.ndarray,
                           filenames: Optional[List[str]] = None,
                           save_path: Optional[str] = None):
    """
    Visualize latent space representations
    
    Args:
        latent_vectors: (N, latent_dim) array of encoded representations
        filenames: Optional list of filenames for labeling
        save_path: Optional path to save figure
    """
    n_samples, latent_dim = latent_vectors.shape
    
    if latent_dim == 2:
        # Direct 2D visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1],
                            c=range(n_samples), cmap='viridis', s=100)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Latent Space (2D)')
        plt.colorbar(scatter, label='Sample Index')
        plt.grid(True, alpha=0.3)
        
        if filenames is not None:
            for i, txt in enumerate(filenames):
                plt.annotate(txt, (latent_vectors[i, 0], latent_vectors[i, 1]),
                           fontsize=8, alpha=0.7)
        
    elif latent_dim >= 3:
        # Use PCA for dimensionality reduction
        fig = plt.figure(figsize=(15, 5))
        
        # 2D projection
        ax1 = fig.add_subplot(121)
        scatter = ax1.scatter(latent_vectors[:, 0], latent_vectors[:, 1],
                            c=range(n_samples), cmap='viridis', s=100)
        ax1.set_xlabel('Latent Dimension 1')
        ax1.set_ylabel('Latent Dimension 2')
        ax1.set_title('Latent Space (First 2 Dims)')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Sample Index')
        
        # 3D projection
        ax2 = fig.add_subplot(122, projection='3d')
        scatter = ax2.scatter(latent_vectors[:, 0], latent_vectors[:, 1], latent_vectors[:, 2],
                            c=range(n_samples), cmap='viridis', s=100)
        ax2.set_xlabel('Latent Dimension 1')
        ax2.set_ylabel('Latent Dimension 2')
        ax2.set_zlabel('Latent Dimension 3')
        ax2.set_title('Latent Space (First 3 Dims)')
        
        plt.tight_layout()
    else:
        # 1D visualization
        plt.figure(figsize=(12, 4))
        plt.scatter(latent_vectors[:, 0], np.zeros_like(latent_vectors[:, 0]),
                   c=range(n_samples), cmap='viridis', s=100)
        plt.xlabel('Latent Dimension 1')
        plt.title('Latent Space (1D)')
        plt.colorbar(label='Sample Index')
        plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_reconstruction_grid(originals: List, 
                                  reconstructions: List,
                                  n_samples: int = 4,
                                  save_path: Optional[str] = None):
    """
    Create a grid visualization comparing original and reconstructed meshes
    Note: This saves renders, doesn't show interactive 3D
    
    Args:
        originals: List of original meshes
        reconstructions: List of reconstructed meshes
        n_samples: Number of samples to show
        save_path: Optional path to save figure
    """
    n_samples = min(n_samples, len(originals))
    
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
    
    for i in range(n_samples):
        # Render original
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(originals[i])
        vis.update_geometry(originals[i])
        vis.poll_events()
        vis.update_renderer()
        img_orig = np.asarray(vis.capture_screen_float_buffer())
        vis.destroy_window()
        
        # Render reconstruction
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(reconstructions[i])
        vis.update_geometry(reconstructions[i])
        vis.poll_events()
        vis.update_renderer()
        img_recon = np.asarray(vis.capture_screen_float_buffer())
        vis.destroy_window()
        
        if n_samples == 1:
            axes[0].imshow(img_orig)
            axes[0].set_title('Original')
            axes[0].axis('off')
            axes[1].imshow(img_recon)
            axes[1].set_title('Reconstructed')
            axes[1].axis('off')
        else:
            axes[0, i].imshow(img_orig)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            axes[1, i].imshow(img_recon)
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    