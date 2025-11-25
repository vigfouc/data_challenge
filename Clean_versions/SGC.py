#Spectral Graph Compression
import torch
import scipy
from scipy.sparse.linalg import eigsh
import numpy as np
import matplotlib.pyplot as plt

class SGC_model():
    def __init__(self):
        self.latent_dim = None
        self.eigenvalues = None
        self.eigenvectors = None


    def get_laplacian_matrix(self, mesh):
        num_nodes = mesh.x.shape[0]
        edge_index = mesh.edge_index

        # Create adjacency matrix
        adj = scipy.sparse.coo_matrix((torch.ones(edge_index.shape[1]), (edge_index[0].numpy(), edge_index[1].numpy())), shape=(num_nodes, num_nodes))
        adj = adj + adj.T  # Make it symmetric
        adj.setdiag(0)  # Remove self-loops
        adj.eliminate_zeros()

        # Degree matrix
        degrees = np.array(adj.sum(axis=1)).flatten()
        D = scipy.sparse.diags(degrees)

        L = D - adj
        return L
    
    def compute_eigendecomposition(self, L, k=50):
        eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        return eigenvalues, eigenvectors
    
    def spectral_embedding(self, x):
        if self.eigenvectors is None:
            raise ValueError("Eigendecomposition not computed. Call compute_eigendecomposition() first.")
        U = torch.tensor(self.eigenvectors, dtype=torch.float, device=x.device)
        return torch.matmul(U.T, x)
    
    def inverse_spectral_embedding(self, z):
        if self.eigenvectors is None:
            raise ValueError("Eigendecomposition not computed. Call compute_eigendecomposition() first.")
        U = torch.tensor(self.eigenvectors, dtype=torch.float, device=z.device)
        return torch.matmul(U, z) 
    
    def forward(self, x):
        z = self.spectral_embedding(x)
        x_reconstructed = self.inverse_spectral_embedding(z)
        return x_reconstructed
    
    def compute_error(self, X_reconstructed, X_original):
        return torch.norm(X_reconstructed - X_original) / (X_original.shape[0] * X_original.shape[1])
    
    def fit(self, dataset, index_ref=0, latent_dim=500, verbose=False):
        self.latent_dim = latent_dim
        mesh_ref = dataset[index_ref]
        L = self.get_laplacian_matrix(mesh_ref)
        self.compute_eigendecomposition(L, k=self.latent_dim)
        for i, data in enumerate(dataset):
            x = data.x
            x_reconstructed = self.forward(x)
            error = self.compute_error(x_reconstructed, x)
            if verbose:
                print(f"Mesh {i} - Reconstruction Error: {error.item():.4f}")
    
    def x_error_on_autoencoding(self, mesh, max_components=500):
        errors = []
        L = self.get_laplacian_matrix(mesh)
        eigenvalues, eigenvectors = self.compute_eigendecomposition(L, k=max_components)
        for k in range(1, max_components + 1):            
            U = torch.tensor(eigenvectors[:, :k], dtype=torch.float, device=mesh.x.device)
            z = torch.matmul(U.T, mesh.x)
            x_reconstructed = torch.matmul(U, z)    
            error = self.compute_error(x_reconstructed, mesh.x)
            errors.append(error.item())
        return errors
    
    def filter_dataset(self, dataset):
        for i, data in enumerate(dataset):
            x = data.x
            z = self.spectral_embedding(x)
            x_filtered = self.inverse_spectral_embedding(z)
            data.x = x_filtered