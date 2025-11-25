import os
import copy
import numpy as np
import torch
import trimesh
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt

# ============================================================================
# 1. LOAD MESHES
# ============================================================================
def load_meshes(data_path: str, normalize: bool = True) -> List[trimesh.Trimesh]:
    """
    Charge tous les fichiers .obj du dossier et retourne une liste de trimesh.Trimesh.
    """
    files = sorted([f for f in os.listdir(data_path) if f.endswith(".obj")])
    if not files:
        raise RuntimeError(f"Aucun fichier .obj trouvé dans {data_path}")
    
    meshes = []
    ref_faces = None
    for f in files:
        mesh = trimesh.load(os.path.join(data_path, f), process=False)
        if normalize:
            verts = mesh.vertices
            centroid = verts.mean(axis=0)
            verts = verts - centroid
            verts /= np.linalg.norm(verts, axis=1).max()
            mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)
        
        if ref_faces is None:
            ref_faces = mesh.faces
        elif not np.array_equal(mesh.faces, ref_faces):
            raise ValueError(f"Topologie différente pour {f}")
        meshes.append(mesh)
    
    print(f"Loaded {len(meshes)} meshes from {data_path}")
    return meshes

# ============================================================================
# 2. CREATE SPLITS
# ============================================================================
def create_splits(meshes: List[trimesh.Trimesh],
                  split_ratios: Tuple[float,float,float]=(0.7,0.15,0.15),
                  random_state: int = 42) -> Tuple[List[trimesh.Trimesh], List[trimesh.Trimesh], List[trimesh.Trimesh]]:
    """
    Crée des splits train/val/test aléatoires.
    """
    N = len(meshes)
    indices = np.arange(N)
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)
    
    n_train = int(N * split_ratios[0])
    n_val = int(N * split_ratios[1])
    
    train = [meshes[i] for i in indices[:n_train]]
    val = [meshes[i] for i in indices[n_train:n_train+n_val]]
    test = [meshes[i] for i in indices[n_train+n_val:]]
    
    print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test

# ============================================================================
# 3. DETECTION POINTS ANATOMIQUES
# ============================================================================
def detect_anatomical_points(mesh_list: list, n_clusters: int = 5, random_state: int = 42) -> np.ndarray:
    """
    Détecte des points anatomiques via K-means sur les vertices de tous les meshes.
    """
    all_vertices = np.vstack([m.vertices for m in mesh_list])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(all_vertices)
    return kmeans.cluster_centers_

def compute_region_centers(mesh, labels, region_map, n_centers_per_region=1):
    """
    Calcule automatiquement les centres anatomiques pour chaque région.
    Peut renvoyer 1 ou plusieurs centres (k-means).
    """
    region_centers = {}

    for region_name, label_idx in region_map.items():
        mask = labels == label_idx
        pts = mesh.vertices[mask]

        if len(pts) == 0:
            region_centers[region_name] = []
            continue

        if n_centers_per_region == 1:
            region_centers[region_name] = [pts.mean(axis=0)]
        else:
            k = min(n_centers_per_region, len(pts))
            km = KMeans(n_clusters=k, n_init='auto').fit(pts)
            region_centers[region_name] = list(km.cluster_centers_)

    return region_centers

def color_mesh_by_centroids(mesh: trimesh.Trimesh, centroids: np.ndarray, random_state: int = 42) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Retourne un mesh coloré selon les centroïdes donnés et les labels de chaque vertex.
    Colors are reproducible by random_state.
    """
    vertices = mesh.vertices
    n_clusters = centroids.shape[0]

    dists = np.linalg.norm(vertices[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)

    rng = np.random.default_rng(random_state)
    colors = rng.integers(0, 256, size=(n_clusters, 3), dtype=np.uint8)
    colors = np.hstack([colors, 255*np.ones((n_clusters,1),dtype=np.uint8)])  # ajouter alpha
    vertex_colors = colors[labels]

    mesh_colored = mesh.copy()
    mesh_colored.visual.vertex_colors = vertex_colors
    return mesh_colored, labels

def plot_cluster_legend(cluster_colors: np.ndarray, n_cols: int = 5, title: str = "Clusters"):
    """
    Affiche un graphe représentant les couleurs des clusters avec une légende lisible.
    cluster_colors: (n_clusters, 3 or 4) in [0,1] or [0,255] range. If 0-255, normalize.
    """
    if cluster_colors.dtype != float:
        cluster_colors = cluster_colors.astype(float) / 255.0

    n_clusters = cluster_colors.shape[0]
    rows = int(np.ceil(n_clusters / n_cols))

    fig, ax = plt.subplots(figsize=(n_cols*1.5, rows*1.2))
    x = np.arange(n_clusters)
    ax.bar(x, 1, color=cluster_colors)
    
    wrapped_labels = [f"Cluster {i}" for i in range(n_clusters)]
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_labels, rotation=45, ha='right')
    ax.set_yticks([])
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def merge_clusters(mesh: trimesh.Trimesh, labels: np.ndarray, merge_list: list, random_state: int = 42):
    """
    Fusionne certaines classes et recolore le mesh.

    merge_list: list whose i-th element is the list of old-cluster-indices to merge into new index i.
                Empty sublists are allowed and represent reserved indices with no merged labels.
    Returns mesh_merged, new_labels
    """
    new_labels = labels.copy().astype(int)
    n_reserved = len(merge_list)

    # mapping old_label -> new_label (based on position in merge_list)
    mapping = {}
    for new_idx, group in enumerate(merge_list):
        if not group:
            continue
        for old_idx in group:
            mapping[int(old_idx)] = new_idx

    # Apply mapping for merged labels
    for i in range(len(new_labels)):
        lab = int(new_labels[i])
        if lab in mapping:
            new_labels[i] = mapping[lab]

    # Remap remaining (unmapped) old labels to consecutive indices starting at n_reserved
    unmapped_old = sorted(set(int(l) for l in labels) - set(mapping.keys()))
    next_idx = n_reserved
    remap_rest = {}
    for old in unmapped_old:
        remap_rest[old] = next_idx
        next_idx += 1

    for i in range(len(new_labels)):
        lab = int(new_labels[i])
        if lab not in mapping:
            # if it was not mapped and corresponds to an original old label, remap
            if lab in remap_rest:
                new_labels[i] = remap_rest[lab]
            else:
                # if label already equals one of merged indices (0..n_reserved-1) leave it
                pass

    # Color generation for present labels
    unique_labels = np.unique(new_labels)
    rng = np.random.default_rng(random_state)
    colors = rng.integers(0, 256, size=(len(unique_labels), 3), dtype=np.uint8)
    colors = np.hstack([colors, 255*np.ones((len(unique_labels), 1), dtype=np.uint8)])
    label_to_color = {lab: colors[i] for i, lab in enumerate(unique_labels)}
    vertex_colors = np.array([label_to_color[int(lab)] for lab in new_labels])

    mesh_merged = mesh.copy()
    mesh_merged.visual.vertex_colors = vertex_colors

    return mesh_merged, new_labels

# ===========================
# Déformations locales (no gauss "spikes")
# ===========================
# ===========================
# Déformations locales avec mask
# ===========================

def local_smooth_scale(mesh: trimesh.Trimesh, center: np.ndarray, radius: float, factor: float, mask: Optional[np.ndarray] = None) -> trimesh.Trimesh:
    m = mesh.copy()
    verts = m.vertices
    if mask is None:
        mask = np.ones(len(verts), dtype=bool)
    diff = verts[mask] - center
    dist = np.linalg.norm(diff, axis=1)
    weight = 0.5 * (1 + np.cos(np.pi * dist / radius))
    scale = 1.0 + (factor - 1.0) * weight[:, None]
    m.vertices[mask] = center + diff * scale
    return m

def local_scaling(mesh: trimesh.Trimesh, center: np.ndarray, radius: float, scale_factor: float, mask: Optional[np.ndarray] = None) -> trimesh.Trimesh:
    m = mesh.copy()
    verts = m.vertices
    if mask is None:
        mask = np.ones(len(verts), dtype=bool)
    diff = verts[mask] - center
    dists = np.linalg.norm(diff, axis=1)
    weights = np.exp(-(dists / radius) ** 2)
    m.vertices[mask] = center + diff * (1 + (scale_factor - 1) * weights[:, None])
    return m

def local_twist(mesh: trimesh.Trimesh, axis_point: np.ndarray, axis_dir: np.ndarray, radius: float, max_angle: float, mask: Optional[np.ndarray] = None) -> trimesh.Trimesh:
    m = mesh.copy()
    verts = m.vertices
    if mask is None:
        mask = np.ones(len(verts), dtype=bool)
    axis_dir = np.asarray(axis_dir, dtype=float)
    axis_norm = np.linalg.norm(axis_dir)
    axis_dir = axis_dir / axis_norm if axis_norm > 0 else np.array([0.0, 0.0, 1.0])

    rel = verts[mask] - axis_point
    t = rel @ axis_dir
    max_t = np.max(np.abs(t)) if np.max(np.abs(t)) > 0 else 1.0
    perp = rel - np.outer(t, axis_dir)
    radial_dist = np.linalg.norm(perp, axis=1)
    radial_weight = np.exp(-(radial_dist / (radius + 1e-12)) ** 2)
    angles = radial_weight * (t / max_t) * max_angle

    u = axis_dir
    tmp = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(u, tmp)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    v = np.cross(u, tmp)
    v /= np.linalg.norm(v) if np.linalg.norm(v) > 0 else 1.0
    w = np.cross(u, v)

    x = rel @ v
    y = rel @ w
    z = rel @ u

    x2 = x * np.cos(angles) - y * np.sin(angles)
    y2 = x * np.sin(angles) + y * np.cos(angles)
    z2 = z

    m.vertices[mask] = axis_point + np.outer(x2, v) + np.outer(y2, w) + np.outer(z2, u)
    return m

def local_bend(mesh: trimesh.Trimesh, center: np.ndarray, radius: float, intensity: float, axis: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None) -> trimesh.Trimesh:
    m = mesh.copy()
    verts = m.vertices
    if mask is None:
        mask = np.ones(len(verts), dtype=bool)
    diff = verts[mask] - center
    dist = np.linalg.norm(diff, axis=1)
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0])
    axis = axis / np.linalg.norm(axis)
    radial = diff - np.outer(diff @ axis, axis)
    weight = (1.0 - dist / radius)[:, None]
    deformation = intensity * weight * radial
    m.vertices[mask] += deformation
    return m

# ===========================
# Déformations par région
# ===========================
def deform_mesh_by_regions(mesh: trimesh.Trimesh,
                           labels: np.ndarray,
                           region_deformations: dict,
                           region_map: dict,
                           region_centers: dict,
                           random_state: int = 42) -> trimesh.Trimesh:
    """
    Applique TOUTES les déformations (scaling, twist, smooth_scaling, bend)
    à TOUTES les régions, pour chaque centre.
    """
    rng = np.random.default_rng(random_state)
    m = mesh.copy()

    for region_name, label_idx in region_map.items():
        mask = labels == label_idx
        if not np.any(mask):
            continue
        centers = region_centers.get(region_name, [])
        if len(centers) == 0:
            continue

        # paramètres région spécifiques
        params = region_deformations.get(region_name, {})

        for center in centers:

            # Rayon avec petite variance
            base_radius = params.get("radius", 0.05)
            radius = base_radius * rng.uniform(0.9, 1.1)

            # ---------------------------
            # 1) SCALING
            # ---------------------------
            factor = params.get("factor", 1.0) * rng.uniform(0.9, 1.1)
            m = local_scaling(m, center, radius, factor, mask=mask)

            # ---------------------------
            # 2) SMOOTH SCALING
            # ---------------------------
            sm_factor = params.get("factor", 1.0) * rng.uniform(0.9, 1.1)
            m = local_smooth_scale(m, center, radius, sm_factor, mask=mask)

            # ---------------------------
            # 3) TWIST
            # ---------------------------
            angle = params.get("angle", np.pi / 20) * rng.uniform(0.9, 1.1)

            axis_twist = rng.normal(size=3)
            axis_twist /= np.linalg.norm(axis_twist) + 1e-9

            m = local_twist(m, center, axis_twist, radius, angle, mask=mask)

            # ---------------------------
            # 4) BEND
            # ---------------------------
            intensity = params.get("intensity", 0.01) * rng.uniform(0.9, 1.1)

            axis_bend = rng.normal(size=3)
            axis_bend /= np.linalg.norm(axis_bend) + 1e-9

            m = local_bend(m, center, radius, intensity, axis=axis_bend, mask=mask)

    return m


# ============================================================================
# 5. AUGMENTATION MESHES (non-linear local augmentations)
# ============================================================================
def augment_meshes(mesh_list: list,
                   labels_list: list,
                   region_deformations: dict,
                   region_map: dict,
                   num_augmented: int = 5,
                   n_centers_per_region: int = 1,
                   random_state: int = 42):
    """
    Génère des variantes de meshes en appliquant deform_mesh_by_regions
    avec randomisation des paramètres de déformation.
    """
    rng = np.random.default_rng(random_state)
    augmented = []

    for mesh, labels in zip(mesh_list, labels_list):

        # centres pour ce mesh
        region_centers = compute_region_centers(
            mesh, labels, region_map, n_centers_per_region=n_centers_per_region
        )

        # Dans augment_meshes, pas de randomisation des paramètres :
        for _ in range(num_augmented):
            new_mesh = deform_mesh_by_regions(
                mesh,
                labels,
                region_deformations,   # <-- passer directement sans modification
                region_map,
                region_centers=region_centers,
                random_state=int(rng.integers(1e9))
            )
            augmented.append(new_mesh)

    return augmented

# ============================================================================
# 6. CONVERSION MESH -> TENSORS / MATRICES / GRAPH
# ============================================================================
def meshes_to_data(mesh_list: List[trimesh.Trimesh], mode: str = "flat") -> List:
    data_out = []
    if mode == "flat":
        for m in mesh_list:
            data_out.append(torch.FloatTensor(m.vertices.flatten()))
    elif mode == "matrix":
        for m in mesh_list:
            data_out.append(torch.FloatTensor(m.vertices))
    elif mode == "graph":
        faces = mesh_list[0].faces
        edges = []
        for (i,j,k) in faces:
            edges += [(i,j),(j,k),(k,i)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)
        for m in mesh_list:
            data_out.append(Data(x=torch.FloatTensor(m.vertices),
                                 edge_index=edge_index,
                                 faces=torch.LongTensor(faces)))
    else:
        raise ValueError("mode doit être 'flat', 'matrix' ou 'graph'")
    return data_out

def data_to_mesh(data_list: List, mode: str = "flat", faces: torch.Tensor = None) -> List[trimesh.Trimesh]:
    """
    Reconstruit des meshes à partir de tenseurs PyTorch.
    
    Args:
        data_list: liste de tenseurs (FloatTensor)
        mode: 'flat', 'matrix', ou 'graph' (doit correspondre à meshes_to_data)
        faces: nécessaire si mode='flat' ou 'matrix' pour reconstruire les faces
    Returns:
        List[trimesh.Trimesh]
    """
    meshes_out = []

    for d in data_list:
        if mode == "flat":
            verts = d.view(-1, 3).cpu().numpy()
        elif mode == "matrix":
            verts = d.cpu().numpy()
        elif mode == "graph":
            verts = d.x.cpu().numpy()
            if faces is None and hasattr(d, 'faces'):
                faces = d.faces.cpu().numpy()
        else:
            raise ValueError("mode doit être 'flat', 'matrix' ou 'graph'")

        if faces is None:
            raise ValueError("faces doit être fourni pour reconstruire le mesh")
        
        meshes_out.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))

    return meshes_out

# ============================================================================
# 7. CREATE DATALOADERS
# ============================================================================
def create_data_loaders(formatted_datasets: Dict[str, List[Union[torch.Tensor, Data]]],
                        batch_size: int = 4,
                        shuffle: bool = True,
                        mode: str = "flat",
                        num_workers: int = 0) -> Dict[str, DataLoader]:
    loaders = {}
    for split in ['train','val','test']:
        if split not in formatted_datasets:
            continue
        data_list = formatted_datasets[split]

        if mode in ['flat','matrix']:
            if isinstance(data_list[0], torch.Tensor):
                tensor_data = torch.stack(data_list)
                dataset = TensorDataset(tensor_data)
            else:
                raise ValueError("Pour mode 'flat' ou 'matrix', les données doivent être des torch.Tensor")
        elif mode == 'graph':
            dataset = data_list
        else:
            raise ValueError("mode doit être 'flat','matrix' ou 'graph'")

        if mode == 'graph':
            loaders[split] = PyGDataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
        else:
            loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=num_workers)
    return loaders

# ============================================================================
# 8. RECONSTRUCTION
# ============================================================================
def reconstruct_vertices(flat_vertices: np.ndarray, dim: int = 3) -> np.ndarray:
    N, D = flat_vertices.shape
    V = D // dim
    return flat_vertices.reshape(N, V, dim)

# ============================================================================
# MAIN EXEMPLE D'USAGE
# ============================================================================
if __name__ == "__main__":
    data_path = "data"

    # -------------------------------------------------------
    print("=== Chargement des meshes ===")
    # -------------------------------------------------------
    meshes = load_meshes(data_path)
    print(f"→ {len(meshes)} meshes chargés")
    print(f"→ {meshes[0].vertices.shape[0]} vertices, {meshes[0].faces.shape[0]} faces")
    print()

    # -------------------------------------------------------
    print("=== Création des splits ===")
    # -------------------------------------------------------
    train_mesh, val_mesh, test_mesh = create_splits(meshes, split_ratios=(0.6,0.2,0.2))
    print(f"Train: {len(train_mesh)}  |  Val: {len(val_mesh)}  |  Test: {len(test_mesh)}")
    print()

    # -------------------------------------------------------
    print("=== Détection des points anatomiques globaux ===")
    # -------------------------------------------------------
    centroids = detect_anatomical_points(meshes, n_clusters=7, random_state=42)
    labeled_mesh, example_labels = color_mesh_by_centroids(meshes[0], centroids)
    print("Centroids shape:", centroids.shape)
    print()

    # -------------------------------------------------------
    print("=== Application des labels à tous les meshes ===")
    # -------------------------------------------------------

    train_labels_list = [color_mesh_by_centroids(m, centroids)[1] for m in train_mesh]
    val_labels_list   = [color_mesh_by_centroids(m, centroids)[1] for m in val_mesh]
    test_labels_list  = [color_mesh_by_centroids(m, centroids)[1] for m in test_mesh]
    print("Labels générés pour train/val/test.")
    print()

    # ==============================================================   
    # PARAMÈTRES DES RÉGIONS
    # ==============================================================
    region_map = {
        "head":                       0,
        "lesser_trochanter":          1,
        "greater_trochanter":         2,
        "body":                       3,
        "popliteal_patellar_surface": 4,
        "medial_condyle":             5,
        "lateral_condyle":            6,
    }

    region_deformations = {
        "head":                       {"type": "scaling", "factor": 1.02, "radius": 0.2},
        "lesser_trochanter":          {"type": "scaling", "factor": 1.0, "radius": 0.08},
        "greater_trochanter":         {"type": "scaling", "factor": 1.02, "radius": 0.1},
        "body":                       {"type": "bend",    "intensity": 0.10, "radius": 0.40},
        "popliteal_patellar_surface": {"type": "bend",    "intensity": 0.05, "radius": 0.20},
        "medial_condyle":             {"type": "scaling", "factor": 1.02, "radius": 0.10},
        "lateral_condyle":            {"type": "scaling", "factor": 1.02, "radius": 0.10},
    }

    # -------------------------------------------------------
    print("=== Augmentation des meshes (train) ===")
    # -------------------------------------------------------
    train_aug = augment_meshes(
        train_mesh,
        train_labels_list,
        region_deformations,
        region_map,
        num_augmented=10,
        random_state=42
    )
    print(f"→ {len(train_aug)} meshes augmentés")
    print()

    # -------------------------------------------------------
    print("=== Augmentation des meshes (val) ===")
    # -------------------------------------------------------
    val_aug = augment_meshes(
        val_mesh,
        val_labels_list,
        region_deformations,
        region_map,
        num_augmented=5,
        random_state=42
    )
    print(f"→ {len(val_aug)} meshes augmentés")
    print()

    # -------------------------------------------------------
    print("=== Conversion en flat tensors ===")
    # -------------------------------------------------------
    X_train_flat = meshes_to_data(train_aug, mode="flat")
    X_val_flat   = meshes_to_data(val_aug, mode="flat")
    X_test_flat  = meshes_to_data(test_mesh, mode="flat")

    X_train_flat_np = np.array([t.numpy() for t in X_train_flat])
    print("Train flat shape:", X_train_flat_np.shape)
    print()

    # -------------------------------------------------------
    print("=== DataLoaders (flat) ===")
    # -------------------------------------------------------
    datasets_flat = {'train': X_train_flat, 'val': X_val_flat, 'test': X_test_flat}
    loaders_flat  = create_data_loaders(datasets_flat, batch_size=8, mode="flat")
    print("DataLoaders flat OK.")
    print()

    # -------------------------------------------------------
    print("=== Reconstruction vertices (vérification) ===")
    # -------------------------------------------------------
    reconstructed = reconstruct_vertices(X_train_flat_np)
    print("Shape reconstructed:", reconstructed.shape)
    print()

    # -------------------------------------------------------
    print("=== Conversion graph PyG ===")
    # -------------------------------------------------------
    X_train_graph = meshes_to_data(train_aug, mode="graph")
    X_val_graph   = meshes_to_data(val_aug,   mode="graph")
    X_test_graph  = meshes_to_data(test_mesh, mode="graph")

    print("Noeuds train:", X_train_graph[0].x.shape[0])
    print("Arêtes train:", X_train_graph[0].edge_index.shape[1])
    print()

    # -------------------------------------------------------
    print("=== DataLoaders PyG ===")
    # -------------------------------------------------------
    datasets_graph = {
        'train': X_train_graph,
        'val':   X_val_graph,
        'test':  X_test_graph
    }
    loaders_graph = create_data_loaders(datasets_graph, batch_size=4, mode="graph")
    print("DataLoaders graph OK.")
    print()

    print("=== Pipeline terminé ===")
