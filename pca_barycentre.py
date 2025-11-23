import numpy as np
import matplotlib.pyplot as plt
import trimesh
from Iterative_PCA_Model import Iterative_PCA
import Data_loader

# Load mesh data
def load_mesh_data():
    mesh_data = Data_loader.MeshDataset()
    mesh_data.load_data()
    X_train = mesh_data.get_all_vertices(mesh_data.dataset_train)
    X_val = mesh_data.get_all_vertices(mesh_data.dataset_val)
    X_test = mesh_data.get_all_vertices(mesh_data.dataset_test)
    return mesh_data, X_train, X_val, X_test

# Augmentation: add barycenter distance as 4th dimension
def add_barycenter_distance(mat_vertices):
    barycenters = np.mean(mat_vertices, axis=1, keepdims=True)  # (N, 1, 3)
    distances = np.linalg.norm(mat_vertices - barycenters, axis=2)  # (N, M)
    mat_vertices_aug = np.concatenate([mat_vertices, distances[..., None]], axis=2)
    return mat_vertices_aug

# Explained variance ratio utility
def explained_variance_ratio_from_model(pca_model, X):
    X_centered = X - np.mean(X, axis=0)
    total_var = np.var(X_centered, axis=0).sum()
    X_proj = pca_model.transform(X)
    var_comps = np.var(X_proj, axis=0)
    return var_comps / total_var

if __name__ == "__main__":
    random_state = 42
    n_components = 7
    np.random.seed(random_state)

    mesh_data, X_train, X_val, X_test = load_mesh_data()

    # Augment data with barycenter distance
    X_train_bary = add_barycenter_distance(X_train)
    X_val_bary = add_barycenter_distance(X_val)
    X_test_bary = add_barycenter_distance(X_test)

    # Flatten for PCA
    X_train_bary_flat = mesh_data.flatten_vertices(X_train_bary)
    X_val_bary_flat = mesh_data.flatten_vertices(X_val_bary)
    X_test_bary_flat = mesh_data.flatten_vertices(X_test_bary)

    # Fit PCA
    pca_model_bary = Iterative_PCA(n_components=n_components, random_state=random_state)
    pca_model_bary.fit_filled(X_train_bary_flat)

    # Explained variance plot
    explained_var_bary = explained_variance_ratio_from_model(pca_model_bary, X_train_bary_flat)
    plt.plot(np.arange(1, len(explained_var_bary)+1), np.cumsum(explained_var_bary), label='PCA barycentre')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title('Explained Variance - PCA Barycentre')
    plt.legend()
    plt.show()

    # Reconstruction
    X_test_bary_flat_reconstructed = pca_model_bary.inverse_transform(pca_model_bary.transform(X_test_bary_flat))
    X_test_bary_reconstructed = mesh_data.reconstruct_vertices_from_flat_vertices(X_test_bary_flat_reconstructed, augmented_dim=1)
    X_test_reconstructed_bary = X_test_bary_reconstructed[:, :, :3]  # Remove barycentre column

    # Compute errors
    errors_bary = []
    for i in range(len(X_test_reconstructed_bary)):
        errors_bary.append(pca_model_bary.compute_error(X_test_reconstructed_bary[i:i+1], X_test[i:i+1]))
    worst_flat_index_bary = np.argmax(errors_bary)

    # Visualize worst original mesh
    worst_vertices_bary = X_test[worst_flat_index_bary]
    worst_original_mesh_bary = mesh_data.reconstruct_mesh_from_vertices(worst_vertices_bary)
    worst_original_mesh_bary.show()

    # Visualize worst reconstructed mesh
    worst_vertices_reconstructed_bary = X_test_reconstructed_bary[worst_flat_index_bary]
    worst_reconstructed_mesh_bary = mesh_data.reconstruct_mesh_from_vertices(worst_vertices_reconstructed_bary)
    worst_reconstructed_mesh_bary.show()

    # Visualize mesh with error coloring
    errors_bary_vertex = np.linalg.norm(worst_vertices_reconstructed_bary - worst_vertices_bary, axis=1)
    errors_bary_norm = (errors_bary_vertex - errors_bary_vertex.min()) / (errors_bary_vertex.max() - errors_bary_vertex.min())
    cmap = plt.get_cmap("jet")
    colors_bary = (cmap(errors_bary_norm)[:, :3] * 255).astype(np.uint8)
    mesh_error_bary = trimesh.Trimesh(
        vertices=worst_vertices_reconstructed_bary,
        faces=mesh_data.faces,
        vertex_colors=colors_bary
    )
    fig, ax = plt.subplots(figsize=(6, 1))
    norm = plt.Normalize(vmin=errors_bary_vertex.min(), vmax=errors_bary_vertex.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=ax, orientation='horizontal')
    cb.set_label('Reconstruction Error (L2 norm) barycentre')
    plt.show()
    mesh_error_bary.show()
