import numpy as np
import open3d as o3d
import os
import torch
from torch.utils.data import Dataset, DataLoader
import copy
from typing import List, Tuple, Optional
from scipy.ndimage import gaussian_filter1d


def load_femur_meshes(data_path: str):
    """
    Load all femur meshes from directory
    
    Args:
        data_path: Path to directory containing .obj files
        
    Returns:
        meshes: List of Open3D meshes
        filenames: List of filenames
    """
    files = sorted([f for f in os.listdir(data_path) if f.endswith(".obj")])
    meshes = [o3d.io.read_triangle_mesh(os.path.join(data_path, f)) for f in files]
    print(f"Loaded {len(meshes)} femurs from {data_path}")
    return meshes, files


def mesh_to_vector(mesh):
    """Convert mesh vertices to flat vector"""
    vertices = np.asarray(mesh.vertices)
    return vertices.flatten()


def vector_to_mesh(vector, reference_mesh):
    """Convert flat vector back to mesh"""
    new_mesh = o3d.geometry.TriangleMesh()
    num_vertices = len(vector) // 3
    vertices = vector.reshape((num_vertices, 3))
    new_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    new_mesh.triangles = reference_mesh.triangles
    return new_mesh


def normalize_mesh(mesh):
    """
    Normalize mesh: center at origin and scale to unit sphere
    
    Args:
        mesh: Open3D mesh
        
    Returns:
        normalized_mesh: Normalized copy of the mesh
    """
    vertices = np.asarray(mesh.vertices)
    
    # Center at origin
    centroid = vertices.mean(axis=0)
    vertices_centered = vertices - centroid
    
    # Scale to unit sphere
    max_dist = np.max(np.linalg.norm(vertices_centered, axis=1))
    vertices_normalized = vertices_centered / max_dist
    
    normalized_mesh = copy.deepcopy(mesh)
    normalized_mesh.vertices = o3d.utility.Vector3dVector(vertices_normalized)
    
    return normalized_mesh


def build_data_matrix(meshes: List):
    """
    Build data matrix from list of meshes
    
    Args:
        meshes: List of Open3D meshes
        
    Returns:
        data_matrix: (N, 3*num_vertices) array
    """
    vectors = np.array([mesh_to_vector(m) for m in meshes])
    print(f"Data matrix shape: {vectors.shape}")
    return vectors


class MeshAugmenter:
    """Data augmentation techniques for 3D meshes"""
    
    @staticmethod
    def random_rotation(mesh, max_angle: float = 15.0):
        """
        Apply random rotation around each axis
        
        Args:
            mesh: Open3D mesh
            max_angle: Maximum rotation angle in degrees
            
        Returns:
            Augmented mesh
        """
        mesh_aug = copy.deepcopy(mesh)
        angles = np.random.uniform(-max_angle, max_angle, 3) * np.pi / 180
        
        # Rotation matrices
        Rx = mesh_aug.get_rotation_matrix_from_xyz((angles[0], 0, 0))
        Ry = mesh_aug.get_rotation_matrix_from_xyz((0, angles[1], 0))
        Rz = mesh_aug.get_rotation_matrix_from_xyz((0, 0, angles[2]))
        
        R = Rz @ Ry @ Rx
        mesh_aug.rotate(R, center=(0, 0, 0))
        return mesh_aug
    
    @staticmethod
    def random_scale(mesh, scale_range: Tuple[float, float] = (0.95, 1.05)):
        """
        Apply random uniform scaling
        
        Args:
            mesh: Open3D mesh
            scale_range: (min_scale, max_scale) tuple
            
        Returns:
            Augmented mesh
        """
        mesh_aug = copy.deepcopy(mesh)
        scale = np.random.uniform(*scale_range)
        mesh_aug.scale(scale, center=(0, 0, 0))
        return mesh_aug
    
    @staticmethod
    def random_noise(mesh, noise_std: float = 0.01):
        """
        Add Gaussian noise to vertices
        
        Args:
            mesh: Open3D mesh
            noise_std: Standard deviation of Gaussian noise
            
        Returns:
            Augmented mesh
        """
        mesh_aug = copy.deepcopy(mesh)
        vertices = np.asarray(mesh_aug.vertices)
        noise = np.random.normal(0, noise_std, vertices.shape)
        vertices_noisy = vertices + noise
        mesh_aug.vertices = o3d.utility.Vector3dVector(vertices_noisy)
        return mesh_aug
    
    @staticmethod
    def random_jitter(mesh, jitter_std: float = 0.005):
        """
        Apply small random perturbations to vertices
        
        Args:
            mesh: Open3D mesh
            jitter_std: Standard deviation of jitter
            
        Returns:
            Augmented mesh
        """
        mesh_aug = copy.deepcopy(mesh)
        vertices = np.asarray(mesh_aug.vertices)
        jitter = np.random.normal(0, jitter_std, vertices.shape)
        vertices_jittered = vertices + jitter
        mesh_aug.vertices = o3d.utility.Vector3dVector(vertices_jittered)
        return mesh_aug
    
    @staticmethod
    def elastic_deformation(mesh, alpha: float = 0.02, sigma: float = 0.05):
        """
        Apply smooth elastic deformation
        
        Args:
            mesh: Open3D mesh
            alpha: Displacement scaling factor
            sigma: Smoothing factor for Gaussian filter
            
        Returns:
            Augmented mesh
        """
        mesh_aug = copy.deepcopy(mesh)
        vertices = np.asarray(mesh_aug.vertices)
        
        # Generate random displacement field
        displacement = np.random.randn(*vertices.shape) * alpha
        
        # Smooth the displacement using Gaussian kernel
        for i in range(3):
            displacement[:, i] = gaussian_filter1d(
                displacement[:, i], 
                sigma=sigma * len(vertices)
            )
        
        vertices_deformed = vertices + displacement
        mesh_aug.vertices = o3d.utility.Vector3dVector(vertices_deformed)
        return mesh_aug


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class FemurDataset(Dataset):
    """
    PyTorch Dataset for femur meshes with data augmentation
    
    Args:
        meshes: List of Open3D meshes
        augment: Whether to apply data augmentation
        augment_factor: Number of augmented samples per original sample
        normalize: Whether to normalize meshes to unit sphere
        return_mesh: If True, return mesh object; if False, return tensor
        augmentation_config: Dictionary with augmentation parameters
    """
    
    def __init__(self, 
                 meshes: List,
                 augment: bool = False,
                 augment_factor: int = 5,
                 normalize: bool = True,
                 return_mesh: bool = False,
                 augmentation_config: Optional[dict] = None):
        
        self.meshes = meshes
        self.augment = augment
        self.augment_factor = augment_factor if augment else 1
        self.normalize = normalize
        self.return_mesh = return_mesh
        self.augmenter = MeshAugmenter()
        
        # Default augmentation configuration
        self.aug_config = {
            'rotation_prob': 0.7,
            'rotation_max_angle': 10.0,
            'scale_prob': 0.7,
            'scale_range': (0.97, 1.03),
            'jitter_prob': 0.5,
            'jitter_std': 0.003,
            'noise_prob': 0.3,
            'noise_std': 0.005,
            'elastic_prob': 0.2,
            'elastic_alpha': 0.015,
            'elastic_sigma': 0.04
        }
        
        # Update with user config if provided
        if augmentation_config is not None:
            self.aug_config.update(augmentation_config)
        
        # Normalize meshes if requested
        if self.normalize:
            self.meshes = [normalize_mesh(m) for m in self.meshes]
            print(f"Normalized {len(self.meshes)} meshes")
        
        # Store reference mesh for reconstruction
        self.reference_mesh = self.meshes[0]
        
    def __len__(self):
        return len(self.meshes) * self.augment_factor
    
    def __getitem__(self, idx):
        # Get base mesh index
        base_idx = idx % len(self.meshes)
        mesh = copy.deepcopy(self.meshes[base_idx])
        
        # Apply augmentation if enabled and not the first occurrence
        if self.augment and (idx // len(self.meshes)) > 0:
            mesh = self._augment_mesh(mesh)
        
        if self.return_mesh:
            return mesh
        else:
            # Convert to tensor
            vertices = np.asarray(mesh.vertices).flatten()
            return torch.FloatTensor(vertices)
    
    def _augment_mesh(self, mesh):
        """Apply random augmentations to mesh"""
        # Random rotation
        if np.random.rand() < self.aug_config['rotation_prob']:
            mesh = self.augmenter.random_rotation(
                mesh, 
                max_angle=self.aug_config['rotation_max_angle']
            )
        
        # Random scaling
        if np.random.rand() < self.aug_config['scale_prob']:
            mesh = self.augmenter.random_scale(
                mesh,
                scale_range=self.aug_config['scale_range']
            )
        
        # Random jitter
        if np.random.rand() < self.aug_config['jitter_prob']:
            mesh = self.augmenter.random_jitter(
                mesh,
                jitter_std=self.aug_config['jitter_std']
            )
        
        # Random noise
        if np.random.rand() < self.aug_config['noise_prob']:
            mesh = self.augmenter.random_noise(
                mesh,
                noise_std=self.aug_config['noise_std']
            )
        
        # Elastic deformation
        if np.random.rand() < self.aug_config['elastic_prob']:
            mesh = self.augmenter.elastic_deformation(
                mesh,
                alpha=self.aug_config['elastic_alpha'],
                sigma=self.aug_config['elastic_sigma']
            )
        
        return mesh


# ============================================================================
# DATALOADER CREATION
# ============================================================================

def create_dataloaders(meshes: List,
                       batch_size: int = 4,
                       augment: bool = True,
                       augment_factor: int = 5,
                       train_split: float = 0.8,
                       normalize: bool = True,
                       num_workers: int = 0,
                       augmentation_config: Optional[dict] = None):
    """
    Create train and validation dataloaders
    
    Args:
        meshes: List of Open3D meshes
        batch_size: Batch size for training
        augment: Whether to apply augmentation to training set
        augment_factor: Number of augmented samples per original sample
        train_split: Fraction of data for training (rest for validation)
        normalize: Whether to normalize meshes
        num_workers: Number of workers for data loading
        augmentation_config: Custom augmentation configuration
        
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        reference_mesh: Reference mesh for reconstruction
    """
    # Split data
    n_train = int(len(meshes) * train_split)
    train_meshes = meshes[:n_train]
    val_meshes = meshes[n_train:]
    
    if len(val_meshes) == 0:
        print("Warning: No validation samples. Using last training sample for validation.")
        val_meshes = [train_meshes[-1]]
    
    # Create datasets
    train_dataset = FemurDataset(
        train_meshes,
        augment=augment,
        augment_factor=augment_factor,
        normalize=normalize,
        augmentation_config=augmentation_config
    )
    
    val_dataset = FemurDataset(
        val_meshes,
        augment=False,
        normalize=normalize
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    print(f"\nDataLoader Summary:")
    print(f"  Train dataset size: {len(train_dataset)} (original: {len(train_meshes)})")
    print(f"  Val dataset size: {len(val_dataset)} (original: {len(val_meshes)})")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches per epoch: {len(train_loader)}")
    print(f"  Val batches per epoch: {len(val_loader)}")
    
    return train_loader, val_loader, train_dataset.reference_mesh


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

if __name__ == "__main__":
    data_path = "Data"
    
    meshes, filenames = load_femur_meshes(data_path)
    
    custom_aug_config = {
        'rotation_prob': 0.8,
        'rotation_max_angle': 15.0,
        'scale_prob': 0.6,
        'jitter_prob': 0.7
    }
    
    train_loader, val_loader, reference_mesh = create_dataloaders(
        meshes,
        batch_size=4,
        augment=True,
        augment_factor=5,
        train_split=0.8,
        normalize=True,
        augmentation_config=custom_aug_config
    )
    
    print("\nTesting dataloader:")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}: shape = {batch.shape}, dtype = {batch.dtype}")
        if i >= 2:
            break
    
    print("\nCreating augmented samples for visualization...")
    aug_dataset = FemurDataset(
        meshes[:1],
        augment=True,
        augment_factor=5,
        return_mesh=True
    )
    
    augmented_meshes = [aug_dataset[i] for i in range(min(5, len(aug_dataset)))]
    
    # Color each mesh differently
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
    for mesh, color in zip(augmented_meshes, colors):
        mesh.paint_uniform_color(color)
    
    print("Displaying original + 4 augmented versions...")
    o3d.visualization.draw_geometries(
        augmented_meshes,
        window_name="Augmentation Samples"
    )