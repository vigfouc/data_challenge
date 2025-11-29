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

# ==========================================================================
# Mesh processing / augmentation pipeline
# This file has been made coherent with the simplified local deformation
# functions (radial_bump, directional_push, radial_wave, plane_squeeze,
# local_shear, local_scaling, soft_FFD). deform_mesh_by_regions maps region
# deformation "type" strings to those functions.
# ==========================================================================

# ============================================================================
# 1. LOAD MESHES
# ============================================================================
def load_meshes(data_path: str, normalize: bool = True) -> List[trimesh.Trimesh]:
    files = sorted([f for f in os.listdir(data_path) if f.endswith(".obj")])
    if not files:
        raise RuntimeError(f"Aucun fichier .obj trouvé dans {data_path}")

    meshes = []
    ref_faces = None
    for f in files:
        mesh = trimesh.load(os.path.join(data_path, f), process=False)
        if normalize:
            verts = mesh.vertices.copy()
            centroid = verts.mean(axis=0)
            verts = verts - centroid
            verts /= np.linalg.norm(verts, axis=1).max() + 1e-12
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
    all_vertices = np.vstack([m.vertices for m in mesh_list])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(all_vertices)
    return kmeans.cluster_centers_


def compute_region_centers(mesh, labels, region_map, n_centers_per_region=1):
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
    vertices = mesh.vertices
    n_clusters = centroids.shape[0]

    dists = np.linalg.norm(vertices[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)

    rng = np.random.default_rng(random_state)
    colors = rng.integers(0, 256, size=(n_clusters, 3), dtype=np.uint8)
    colors = np.hstack([colors, 255*np.ones((n_clusters,1),dtype=np.uint8)])
    vertex_colors = colors[labels]

    mesh_colored = mesh.copy()
    mesh_colored.visual.vertex_colors = vertex_colors
    return mesh_colored, labels


def plot_cluster_legend(cluster_colors: np.ndarray, n_cols: int = 5, title: str = "Clusters"):
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
    new_labels = labels.copy().astype(int)
    n_reserved = len(merge_list)

    mapping = {}
    for new_idx, group in enumerate(merge_list):
        if not group:
            continue
        for old_idx in group:
            mapping[int(old_idx)] = new_idx

    for i in range(len(new_labels)):
        lab = int(new_labels[i])
        if lab in mapping:
            new_labels[i] = mapping[lab]

    unmapped_old = sorted(set(int(l) for l in labels) - set(mapping.keys()))
    next_idx = n_reserved
    remap_rest = {}
    for old in unmapped_old:
        remap_rest[old] = next_idx
        next_idx += 1

    for i in range(len(new_labels)):
        lab = int(new_labels[i])
        if lab not in mapping:
            if lab in remap_rest:
                new_labels[i] = remap_rest[lab]

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
# LOCAL DEFORMATION FUNCTIONS (Optimized)
# ===========================

def radial_bump_inplace(m, v, mask, rel, w, intensity):
    v[mask] += rel * (w * intensity)[:, None]

def directional_push_inplace(m, v, mask, w, direction, strength):
    v[mask] += direction * (w * strength)[:, None]

def radial_wave_inplace(m, v, mask, rel, w, amplitude, frequency):
    d = np.linalg.norm(rel, axis=1)
    dir_norm = rel / (d[:, None] + 1e-12)
    disp = amplitude * np.sin(frequency * d) * w
    v[mask] += dir_norm * disp[:, None]

def plane_squeeze_inplace(m, v, mask, rel, plane_normal, strength):
    dist = rel @ plane_normal
    w = np.exp(-(dist / np.max([1e-12, dist.max()]) / 1.0)**2)
    v[mask] -= plane_normal * (dist * strength * w)[:, None]

def local_shear_inplace(m, v, mask, rel, shear_axis, shear_dir, amount):
    proj = rel @ shear_axis
    w = np.exp(-(np.linalg.norm(rel, axis=1) / np.max([1e-12, np.linalg.norm(rel, axis=1).max()]))**2)
    v[mask] += shear_dir * (proj * amount * w)[:, None]

def local_scaling_inplace(m, v, mask, rel, scale_factor):
    dist = np.linalg.norm(rel, axis=1)[:, None]
    w = np.exp(-(dist / np.max([1e-12, dist.max()]))**2)
    v[mask][:] = m.vertices[mask] + rel * ((scale_factor - 1) * w)

def soft_FFD_inplace(m, v, mask, offset, w):
    v[mask] += offset * w[:, None]

# ===========================
# APPLY TYPE (Optimized)
# ===========================
def apply_type_inplace(m, typ, center, radius, params, mask, rng):
    v = m.vertices
    rel = v[mask] - center
    if typ == 'scaling':
        local_scaling_inplace(m, v, mask, rel, params.get('factor', 1.0))
    elif typ == 'bump':
        d = np.linalg.norm(rel, axis=1)
        w = np.clip(1 - d / radius, 0, 1)
        radial_bump_inplace(m, v, mask, rel, w, params.get('intensity', 0.02))
    elif typ == 'push':
        direction = params.get('direction', rng.normal(size=3))
        direction /= np.linalg.norm(direction) + 1e-12
        d = np.linalg.norm(rel, axis=1)
        w = np.clip(1 - d / radius, 0, 1)
        directional_push_inplace(m, v, mask, w, direction, params.get('strength', 0.02))
    elif typ == 'wave':
        amplitude = params.get('amplitude', 0.01)
        frequency = params.get('frequency', 10.0)
        d = np.linalg.norm(rel, axis=1)
        w = np.clip(1 - d / radius, 0, 1)
        radial_wave_inplace(m, v, mask, rel, w, amplitude, frequency)
    elif typ in ['squeeze', 'bend']:
        plane_normal = params.get('axis', rng.normal(size=3))
        plane_normal /= np.linalg.norm(plane_normal) + 1e-12
        strength = params.get('intensity', 0.05)
        plane_squeeze_inplace(m, v, mask, rel, plane_normal, strength)
    elif typ == 'shear':
        shear_axis = params.get('shear_axis', np.array([1,0,0]))
        shear_dir = params.get('shear_dir', np.array([0,1,0]))
        amount = params.get('amount', 0.02)
        local_shear_inplace(m, v, mask, rel, shear_axis, shear_dir, amount)
    elif typ == 'ffd':
        offset = params.get('offset', np.zeros(3))
        w = np.exp(-(np.linalg.norm(rel, axis=1) / radius)**2)
        soft_FFD_inplace(m, v, mask, offset, w)
    else:  # fallback
        local_scaling_inplace(m, v, mask, rel, params.get('factor', 1.0))

# ===========================
# DEFORM MESH BY REGIONS (Optimized)
# ===========================
def deform_mesh_by_regions(mesh: trimesh.Trimesh,
                           labels: np.ndarray,
                           region_deformations: Dict[str, dict],
                           region_map: Dict[str, int],
                           region_centers: Dict[str, List[np.ndarray]],
                           random_state: int = 42) -> trimesh.Trimesh:
    rng = np.random.default_rng(random_state)
    m = mesh.copy()
    v = m.vertices

    for region_name, label_idx in region_map.items():
        mask = labels == label_idx
        if not np.any(mask):
            continue
        centers = region_centers.get(region_name, [])
        if len(centers) == 0:
            continue

        deforms_list = region_deformations.get(region_name, {}).get("deforms", [])
        for center in centers:
            for deform in deforms_list:
                base_radius = deform.get('radius', 0.05)
                radius = float(base_radius * rng.uniform(0.8,1.2))
                params_rand = deform.copy()
                for key in ['factor','intensity','strength','amount','amplitude','frequency','shear']:
                    if key in params_rand:
                        params_rand[key] *= float(rng.uniform(0.8,1.2))
                apply_type_inplace(m, deform['type'], center, radius, params_rand, mask, rng)
    return m

# ===========================
# AUGMENT NON-Linear MESHES (Optimized)
# ===========================
def augment_non_linear_meshes(mesh_list, labels_list, region_deformations, region_map,
                   augmentation_factor=5, n_centers_per_region=1, random_state=42):
    rng = np.random.default_rng(random_state)
    augmented = []

    for mesh, labels in zip(mesh_list, labels_list):
        region_centers = compute_region_centers(mesh, labels, region_map, n_centers_per_region=n_centers_per_region)
        for _ in range(augmentation_factor):
            new_mesh = deform_mesh_by_regions(mesh, labels, region_deformations, region_map, region_centers, random_state=int(rng.integers(1e9)))
            augmented.append(new_mesh)
    return augmented

# ===========================
# AUGMENT GEOMETRIC MESHES
# ===========================

def random_rotation(mesh, max_angle: float = 15.0):
    """Rotation aléatoire autour des axes X, Y, Z"""
    mesh_aug = copy.deepcopy(mesh)
    angles = np.random.uniform(-max_angle, max_angle, 3) * np.pi / 180

    # Matrices de rotation
    Rx = trimesh.transformations.rotation_matrix(angles[0], [1, 0, 0])
    Ry = trimesh.transformations.rotation_matrix(angles[1], [0, 1, 0])
    Rz = trimesh.transformations.rotation_matrix(angles[2], [0, 0, 1])
    
    # Combinaison des rotations
    R = trimesh.transformations.concatenate_matrices(Rz, Ry, Rx)
    
    mesh_aug.apply_transform(R)
    return mesh_aug

def random_scale(mesh, scale_range: Tuple[float, float] = (0.95, 1.05)):
    """Mise à l'échelle uniforme aléatoire"""
    mesh_aug = copy.deepcopy(mesh)
    scale = np.random.uniform(*scale_range)
    mesh_aug.apply_scale(scale)
    return mesh_aug

def random_noise(mesh, noise_std: float = 0.01):
    """Ajout de bruit gaussien aux vertices"""
    mesh_aug = copy.deepcopy(mesh)
    vertices = mesh_aug.vertices.copy()
    noise = np.random.normal(0, noise_std, vertices.shape)
    mesh_aug.vertices += noise
    return mesh_aug

def random_jitter(mesh, jitter_std: float = 0.005):
    """Petites perturbations aléatoires aux vertices"""
    mesh_aug = copy.deepcopy(mesh)
    vertices = mesh_aug.vertices.copy()
    jitter = np.random.normal(0, jitter_std, vertices.shape)
    mesh_aug.vertices += jitter
    return mesh_aug

def elastic_deformation(mesh, alpha: float = 0.02, sigma: float = 0.05):
    """Déformation élastique douce"""
    mesh_aug = copy.deepcopy(mesh)
    vertices = mesh_aug.vertices.copy()
    displacement = np.random.randn(*vertices.shape) * alpha
    for i in range(3):
        displacement[:, i] = gaussian_filter1d(displacement[:, i], sigma=sigma * len(vertices))
    mesh_aug.vertices += displacement
    return mesh_aug

# ===========================
# Fonction principale d'augmentation géométrique
# ===========================

def augment_geometric_meshes(mesh, aug_config: Dict):
    """
    Applique de manière aléatoire plusieurs déformations géométriques sur un mesh trimesh
    
    Args:
        mesh: trimesh.Trimesh
        aug_config: dictionnaire contenant les probabilités et paramètres
            {
                'rotation_prob': float,
                'rotation_max_angle': float,
                'scale_prob': float,
                'scale_range': (min_scale, max_scale),
                'jitter_prob': float,
                'jitter_std': float,
                'noise_prob': float,
                'noise_std': float,
                'elastic_prob': float,
                'elastic_alpha': float,
                'elastic_sigma': float
            }
            
    Returns:
        mesh augmentée
    """
    mesh_aug = copy.deepcopy(mesh)
    
    if np.random.rand() < aug_config.get('rotation_prob', 0.0):
        mesh_aug = random_rotation(mesh_aug, max_angle=aug_config.get('rotation_max_angle', 15.0))
        
    if np.random.rand() < aug_config.get('scale_prob', 0.0):
        mesh_aug = random_scale(mesh_aug, scale_range=aug_config.get('scale_range', (0.95, 1.05)))
        
    if np.random.rand() < aug_config.get('jitter_prob', 0.0):
        mesh_aug = random_jitter(mesh_aug, jitter_std=aug_config.get('jitter_std', 0.005))
        
    if np.random.rand() < aug_config.get('noise_prob', 0.0):
        mesh_aug = random_noise(mesh_aug, noise_std=aug_config.get('noise_std', 0.01))
        
    if np.random.rand() < aug_config.get('elastic_prob', 0.0):
        mesh_aug = elastic_deformation(
            mesh_aug, 
            alpha=aug_config.get('elastic_alpha', 0.02),
            sigma=aug_config.get('elastic_sigma', 0.05)
        )
        
    return mesh_aug

# ===========================
# AUGMENT Linear MESHES (PCA)
# ===========================
def augment_linear_meshes(mesh_list, num_gen=5, random_state=42):
    rng = np.random.default_rng(random_state)
    all_vertices = np.array([m.vertices.flatten() for m in mesh_list])
    pca = PCA(n_components=0.999, random_state=random_state)
    pca.fit(all_vertices)

    augmented = []
    for _ in range(num_gen):
        coeffs = rng.normal(scale=0.1, size=pca.n_components_)
        new_flat = pca.mean_ + coeffs @ pca.components_
        new_verts = new_flat.reshape(-1, 3)
        new_mesh = trimesh.Trimesh(vertices=new_verts, faces=mesh_list[0].faces, process=False)
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
    elif mode == "3D":
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
        raise ValueError("mode doit être 'flat', '3D' ou 'graph'")
    return data_out


def data_to_mesh(data_list: List, mode: str = "flat", faces: torch.Tensor = None) -> List[trimesh.Trimesh]:
    meshes_out = []
    for d in data_list:
        if mode == "flat":
            verts = d.view(-1, 3).cpu().numpy()
        elif mode == "3D":
            verts = d.cpu().numpy()
        elif mode == "graph":
            verts = d.x.cpu().numpy()
            if faces is None and hasattr(d, 'faces'):
                faces = d.faces.cpu().numpy()
        else:
            raise ValueError("mode doit être 'flat', '3D' ou 'graph'")

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

        if mode in ['flat','3D']:
            if isinstance(data_list[0], torch.Tensor):
                tensor_data = torch.stack(data_list)
                dataset = TensorDataset(tensor_data)
            else:
                raise ValueError("Pour mode 'flat' ou '3D', les données doivent être des torch.Tensor")
        elif mode == 'graph':
            dataset = data_list
        else:
            raise ValueError("mode doit être 'flat','3D' ou 'graph'")
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

    print("=== Chargement des meshes ===")
    meshes = load_meshes(data_path)
    print(f"→ {len(meshes)} meshes chargés")
    print(f"→ {meshes[0].vertices.shape[0]} vertices, {meshes[0].faces.shape[0]} faces\n")

    print("=== Création des splits ===")
    train_mesh, val_mesh, test_mesh = create_splits(meshes, split_ratios=(0.6,0.2,0.2))
    print(f"Train: {len(train_mesh)}  |  Val: {len(val_mesh)}  |  Test: {len(test_mesh)}\n")

    print("=== Détection des points anatomiques globaux ===")
    centroids = detect_anatomical_points(meshes, n_clusters=7, random_state=42)
    labeled_mesh, example_labels = color_mesh_by_centroids(meshes[0], centroids)
    print("Centroids shape:", centroids.shape, "\n")

    print("=== Application des labels à tous les meshes ===")
    train_labels_list = [color_mesh_by_centroids(m, centroids)[1] for m in train_mesh]
    val_labels_list   = [color_mesh_by_centroids(m, centroids)[1] for m in val_mesh]
    test_labels_list  = [color_mesh_by_centroids(m, centroids)[1] for m in test_mesh]
    print("Labels générés pour train/val/test.\n")

    # ===========================
    # AUGMENTATION LINÉAIRE (PCA)
    # ===========================
    print("=== Augmentation linéaire (PCA) train ===")
    train_aug_lin = augment_linear_meshes(train_mesh, num_gen=5, random_state=42)
    print(f"→ {len(train_aug_lin)} meshes augmentés (linéaire)\n")

    print("=== Augmentation linéaire (PCA) val ===")
    val_aug_lin = augment_linear_meshes(val_mesh, num_gen=3, random_state=42)
    print(f"→ {len(val_aug_lin)} meshes augmentés (linéaire)\n")

    # ===========================
    # AUGMENTATION GÉOMÉTRIQUE / NON-LINÉAIRE SUR LES MESHS PCA
    # ===========================

    # Déformations locales par région
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
        "head":                       {"deforms":[{"type":"scaling","factor":1.02,"radius":0.2}]},
        "lesser_trochanter":          {"deforms":[{"type":"scaling","factor":1.0,"radius":0.08}]},
        "greater_trochanter":         {"deforms":[{"type":"scaling","factor":1.02,"radius":0.1}]},
        "body":                       {"deforms":[{"type":"bend","intensity":0.10,"radius":0.40}]},
        "popliteal_patellar_surface": {"deforms":[{"type":"bend","intensity":0.05,"radius":0.20}]},
        "medial_condyle":             {"deforms":[{"type":"scaling","factor":1.02,"radius":0.10}]},
        "lateral_condyle":            {"deforms":[{"type":"scaling","factor":1.02,"radius":0.10}]},
    }

    # Config géométrique
    aug_config = {
        'rotation_prob': 0.7, 'rotation_max_angle': 15.0,
        'scale_prob': 0.5, 'scale_range': (0.95, 1.05),
        'jitter_prob': 0.5, 'jitter_std': 0.005,
        'noise_prob': 0.5, 'noise_std': 0.01,
        'elastic_prob': 0.3, 'elastic_alpha': 0.02, 'elastic_sigma': 0.05
    }

    print("=== Augmentation non-linéaire sur meshes PCA (train) ===")
    train_aug_nl = augment_non_linear_meshes(
        train_aug_lin,
        train_labels_list,  # labels du train original
        region_deformations,
        region_map,
        augmentation_factor=2,  # nombre d’itérations non-linéaires par mesh PCA
        n_centers_per_region=1,
        random_state=42
    )
    print(f"→ {len(train_aug_nl)} meshes augmentés (non-linéaire)\n")

    print("=== Augmentation géométrique sur meshes PCA (train) ===")
    train_aug_geo = [augment_geometric_meshes(m, aug_config) for m in train_aug_lin]
    print(f"→ {len(train_aug_geo)} meshes augmentés (géométrique)\n")

    # Même chose pour val
    print("=== Augmentation non-linéaire sur meshes PCA (val) ===")
    val_aug_nl = augment_non_linear_meshes(
        val_aug_lin,
        val_labels_list,
        region_deformations,
        region_map,
        augmentation_factor=1,
        n_centers_per_region=1,
        random_state=42
    )
    print(f"→ {len(val_aug_nl)} meshes augmentés (non-linéaire)\n")

    print("=== Augmentation géométrique sur meshes PCA (val) ===")
    val_aug_geo = [augment_geometric_meshes(m, aug_config) for m in val_aug_lin]
    print(f"→ {len(val_aug_geo)} meshes augmentés (géométrique)\n")

    # ===========================
    # COMBINAISON DES AUGMENTATIONS
    # ===========================
    train_all = train_mesh + train_aug_lin + train_aug_nl + train_aug_geo
    val_all   = val_mesh   + val_aug_lin   + val_aug_nl   + val_aug_geo
    test_all  = test_mesh

    print(f"Total train meshes après augmentation: {len(train_all)}")
    print(f"Total val meshes après augmentation: {len(val_all)}")
    print(f"Total test meshes: {len(test_all)}\n")

    # ===========================
    # CONVERSION EN FLAT TENSORS
    # ===========================
    X_train_flat = meshes_to_data(train_all, mode="flat")
    X_val_flat   = meshes_to_data(val_all, mode="flat")
    X_test_flat  = meshes_to_data(test_all, mode="flat")

    X_train_flat_np = np.array([t.numpy() for t in X_train_flat])
    print("Train flat shape:", X_train_flat_np.shape, "\n")

    # ===========================
    # DATALOADERS
    # ===========================
    datasets_flat = {'train': X_train_flat, 'val': X_val_flat, 'test': X_test_flat}
    loaders_flat  = create_data_loaders(datasets_flat, batch_size=8, mode="flat")
    print("DataLoaders flat OK.\n")

    # ===========================
    # CONVERSION EN GRAPH PYG
    # ===========================
    X_train_graph = meshes_to_data(train_all, mode="graph")
    X_val_graph   = meshes_to_data(val_all, mode="graph")
    X_test_graph  = meshes_to_data(test_all, mode="graph")

    print("Noeuds train:", X_train_graph[0].x.shape[0])
    print("Arêtes train:", X_train_graph[0].edge_index.shape[1], "\n")

    datasets_graph = {'train': X_train_graph, 'val': X_val_graph, 'test': X_test_graph}
    loaders_graph = create_data_loaders(datasets_graph, batch_size=4, mode="graph")
    print("DataLoaders graph OK.\n")

    print("=== Pipeline terminé ===")