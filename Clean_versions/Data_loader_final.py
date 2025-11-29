import os
import numpy as np
import torch
import trimesh
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import List, Tuple, Dict, Union
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ============================================================================
# 1. LOAD MESHES
# ============================================================================
def load_meshes(data_path: str, normalize: bool = True) -> List[trimesh.Trimesh]:
    """
    Charge tous les fichiers .obj d'un répertoire et les normalise si demandé.
    
    Args:
        data_path (str): Chemin vers le dossier contenant les meshes .obj.
        normalize (bool): Si True, centre et met à l'échelle les vertices dans [-1,1].
    
    Returns:
        List[trimesh.Trimesh]: Liste des meshes chargés et normalisés.
    
    Raises:
        RuntimeError: Si aucun fichier .obj trouvé.
        ValueError: Si des meshes ont des topologies différentes.
    """
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
    """
    Crée les splits train/val/test à partir de la liste de meshes.
    
    Args:
        meshes (List[trimesh.Trimesh]): Liste des meshes à splitter.
        split_ratios (tuple): Ratio de division (train, val, test).
        random_state (int): Seed pour le shuffle.
    
    Returns:
        Tuple[List[trimesh.Trimesh], List[trimesh.Trimesh], List[trimesh.Trimesh]]: Splits train, val, test.
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
    Détecte les points anatomiques globaux via KMeans sur tous les vertices.
    
    Args:
        mesh_list (list): Liste des meshes.
        n_clusters (int): Nombre de clusters (points anatomiques).
        random_state (int): Seed pour KMeans.
    
    Returns:
        np.ndarray: Coordonnées des centroïdes détectés.
    """
    all_vertices = np.vstack([m.vertices for m in mesh_list])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(all_vertices)
    return kmeans.cluster_centers_


def compute_region_centers(mesh, labels, region_map, n_centers_per_region=1):
    """
    Calcule les centres de chaque région définie dans region_map pour un mesh donné.
    
    Args:
        mesh (trimesh.Trimesh): Mesh à analyser.
        labels (np.ndarray): Labels des vertices.
        region_map (dict): Mapping des noms de régions vers indices de labels.
        n_centers_per_region (int): Nombre de centres à calculer par région.
    
    Returns:
        dict: Clés=nom de région, valeurs=liste des centres (np.ndarray).
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
    Colore chaque vertex du mesh selon le centroïde le plus proche.
    
    Args:
        mesh (trimesh.Trimesh): Mesh à colorer.
        centroids (np.ndarray): Coordonnées des centroïdes.
        random_state (int): Seed pour génération aléatoire des couleurs.
    
    Returns:
        Tuple[trimesh.Trimesh, np.ndarray]: Mesh coloré et labels des vertices.
    """
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
    """
    Affiche une légende graphique des clusters avec leurs couleurs.
    
    Args:
        cluster_colors (np.ndarray): Couleurs des clusters (N,3 ou N,4).
        n_cols (int): Nombre de colonnes pour la visualisation.
        title (str): Titre du plot.
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
    Fusionne certains clusters selon merge_list et recolore le mesh.
    
    Args:
        mesh (trimesh.Trimesh): Mesh à modifier.
        labels (np.ndarray): Labels initiaux des vertices.
        merge_list (list of list): Chaque sous-liste contient les indices de clusters à fusionner.
        random_state (int): Seed pour couleurs aléatoires.
    
    Returns:
        Tuple[trimesh.Trimesh, np.ndarray]: Mesh recoloré et labels mis à jour.
    """
    labels = labels.copy().astype(int)
    rng = np.random.default_rng(random_state)

    # -------------------------------
    # 1. Mapping pour les clusters fusionnés
    # -------------------------------
    mapping = {}
    for new_idx, group in enumerate(merge_list):
        for old_idx in group:
            mapping[old_idx] = new_idx

    # -------------------------------
    # 2. Remap labels existants
    # -------------------------------
    new_labels = np.array([mapping.get(l, l) for l in labels])

    # -------------------------------
    # 3. Re-indexation des labels non fusionnés pour qu'ils soient contigus
    # -------------------------------
    unique_labels = np.unique(new_labels)
    reindex = {old: new for new, old in enumerate(unique_labels)}
    new_labels = np.array([reindex[l] for l in new_labels])

    # -------------------------------
    # 4. Générer les couleurs et appliquer
    # -------------------------------
    n_clusters = len(unique_labels)
    colors = rng.integers(0, 256, size=(n_clusters, 3), dtype=np.uint8)
    colors = np.hstack([colors, 255*np.ones((n_clusters, 1), dtype=np.uint8)])
    vertex_colors = colors[new_labels]

    mesh_merged = mesh.copy()
    mesh_merged.visual.vertex_colors = vertex_colors

    return mesh_merged, new_labels

# ============================================================================
# RADIAL WEIGHTS GPU
# ============================================================================
def radial_weights(rel: torch.Tensor, radius: float, profile="gaussian") -> torch.Tensor:
    if radius <= torch.finfo(rel.dtype).eps:
        return torch.ones(rel.shape[0], device=rel.device)
    d = torch.norm(rel, dim=1)
    if profile == "linear":
        return torch.clamp(1 - d / radius, 0, 1)
    return torch.exp(-(d / (radius * 0.6))**2)


# ============================================================================
# AUGMENTATION NON-LINÉAIRE GPU OPTIMISÉE
# ============================================================================
def augment_non_linear_meshes(mesh_list, labels_list, region_deformations,
                              region_map, augmentation_factor=5,
                              n_centers_per_region=1, device='cuda',
                              random_state=42):
    rng = np.random.default_rng(random_state)
    augmented_meshes = []

    # --- Pré-calcul GPU des centres pour tous les meshes ---
    region_centers_gpu_list = []
    for mesh, labels in zip(mesh_list, labels_list):
        region_centers = compute_region_centers(mesh, labels, region_map,
                                                n_centers_per_region=n_centers_per_region)
        region_centers_gpu = {r: [torch.tensor(c, dtype=torch.float32, device=device) 
                                  for c in cs]
                              for r, cs in region_centers.items()}
        region_centers_gpu_list.append(region_centers_gpu)

    # --- Boucle d'augmentation ---
    for mesh, labels, region_centers_gpu in zip(mesh_list, labels_list, region_centers_gpu_list):
        v = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)
        gen = torch.Generator(device=device).manual_seed(random_state)

        # Préparer les indices de chaque label une seule fois
        mask_dict = {region_name: (labels_t == label_idx).nonzero(as_tuple=True)[0]
                     for region_name, label_idx in region_map.items()}

        for _ in range(augmentation_factor):
            v_aug = v.clone()

            for region_name, indices in mask_dict.items():
                if len(indices) == 0:
                    continue

                centers_list = region_centers_gpu.get(region_name, [])
                if not centers_list:
                    continue

                deforms_list = region_deformations.get(region_name, {}).get("deforms", [])
                for center in centers_list:
                    rel = v_aug[indices] - center

                    for deform in deforms_list:
                        deform_type = deform.get("type", "scaling")
                        if deform_type not in ['scaling', 'bump', 'twist']:
                            continue

                        # Rayon et paramètres aléatoires sur GPU
                        radius = deform.get("radius", 0.05) * (0.8 + 0.45 * torch.rand(1, generator=gen, device=device))
                        params = {}
                        for k, val in deform.items():
                            if k in ["factor", "intensity", "max_angle"]:
                                params[k] = float(val) * (0.85 + 0.3 * torch.rand(1, generator=gen, device=device))
                            elif k in ["axis_point", "axis_dir"]:
                                vec = torch.tensor(val, dtype=torch.float32, device=device)
                                params[k] = vec / (vec.norm() + 1e-12)

                        # --- Appliquer la déformation vectorisée ---
                        if deform_type == 'scaling':
                            w = radial_weights(rel, radius)
                            factor = params.get('factor', 1.0)
                            scaled_rel = rel * (1 + (factor-1) * w[:, None])
                            v_aug[indices] = (v_aug[indices] - rel) + scaled_rel

                        elif deform_type == 'bump':
                            w = radial_weights(rel, radius)
                            intensity = params.get('intensity', 0.05)
                            v_aug[indices] += rel * (w * intensity)[:, None]

                        elif deform_type == 'twist':
                            axis_point = params.get('axis_point', center)
                            axis_dir = params.get('axis_dir', torch.tensor([0.,0.,1.], device=device))
                            axis_dir = axis_dir / (axis_dir.norm() + 1e-12)

                            vertices = v_aug[indices]
                            proj = (vertices - axis_point) @ axis_dir
                            w = radial_weights(vertices - axis_point, radius)
                            max_angle = params.get('max_angle', 0.2)
                            angles = max_angle * proj / (proj.max() + 1e-12) * w

                            cos_theta = torch.cos(angles)[:, None]
                            sin_theta = torch.sin(angles)[:, None]
                            p = vertices - axis_point
                            k = axis_dir
                            rotated = p * cos_theta + torch.linalg.cross(k.expand_as(p), p) * sin_theta \
                                      + k * (p @ k)[:, None] * (1 - cos_theta)
                            v_aug[indices] = axis_point + rotated

            augmented_meshes.append(trimesh.Trimesh(vertices=v_aug.cpu().numpy(),
                                                    faces=mesh.faces,
                                                    process=False))

    return augmented_meshes


# ===========================
# AUGMENT GEOMETRIC MESHES
# ===========================
def augment_geometric_meshes(mesh_list, aug_config, device='cuda'):
    """
    Augmente une liste de meshes sur GPU avec rotation, scaling, jitter, noise, elastic deformation.
    Chaque mesh est augmenté indépendamment.
    """
    meshes_aug = []
    faces = mesh_list[0].faces  # même topologie pour tous les meshes

    for mesh in mesh_list:
        verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)

        # -----------------------------
        # Rotation
        # -----------------------------
        if torch.rand(1, device=device) < aug_config.get('rotation_prob', 0.0):
            angles = (torch.rand(3, device=device) * 2 - 1) * aug_config.get('rotation_max_angle', 15.0) * (torch.pi/180)
            cx, cy, cz = torch.cos(angles)
            sx, sy, sz = torch.sin(angles)
            Rx = torch.tensor([[1,0,0],[0,cx,-sx],[0,sx,cx]], device=device)
            Ry = torch.tensor([[cy,0,sy],[0,1,0],[-sy,0,cy]], device=device)
            Rz = torch.tensor([[cz,-sz,0],[sz,cz,0],[0,0,1]], device=device)
            R = Rz @ Ry @ Rx
            verts = verts @ R.T

        # -----------------------------
        # Scaling
        # -----------------------------
        if torch.rand(1, device=device) < aug_config.get('scale_prob', 0.0):
            scale = torch.empty(1, device=device).uniform_(*aug_config.get('scale_range', (0.95,1.05)))
            verts = verts * scale

        # -----------------------------
        # Jitter
        # -----------------------------
        if torch.rand(1, device=device) < aug_config.get('jitter_prob', 0.0):
            verts = verts + torch.randn_like(verts) * aug_config.get('jitter_std', 0.005)

        # -----------------------------
        # Noise
        # -----------------------------
        if torch.rand(1, device=device) < aug_config.get('noise_prob', 0.0):
            verts = verts + torch.randn_like(verts) * aug_config.get('noise_std', 0.01)

        # -----------------------------
        # Elastic (GPU)
        # -----------------------------
        if aug_config.get('elastic_prob', 0.0) > 0 and torch.rand(1, device=device) < aug_config['elastic_prob']:
            alpha = aug_config.get('elastic_alpha', 0.02)
            sigma = aug_config.get('elastic_sigma', 0.05)
            N = verts.shape[0]

            # bruit initial
            disp = torch.randn_like(verts) * alpha  # [N,3]

            # kernel gaussien
            kernel_size = int(6*sigma*N) | 1  # impair
            x = torch.arange(kernel_size, device=device) - kernel_size // 2
            gauss = torch.exp(-(x.float()**2)/(2*(sigma*N)**2))
            gauss = gauss / gauss.sum()
            gauss_kernel = gauss.view(1, 1, -1).repeat(3, 1, 1)  # [3,1,K]

            # convolution 1D sur chaque canal
            disp = disp.permute(1,0).unsqueeze(0)  # [1,3,N]
            disp = F.pad(disp, (kernel_size//2, kernel_size//2), mode='circular')
            disp = F.conv1d(disp, gauss_kernel, groups=3)
            disp = disp[:, :, :N]  # tronquer si nécessaire
            disp = disp.squeeze(0).permute(1,0)  # [N,3]

            verts = verts + disp


        meshes_aug.append(trimesh.Trimesh(vertices=verts.cpu().numpy(),
                                          faces=faces,
                                          process=False))

    return meshes_aug

# ===========================
# AUGMENT Linear MESHES (PCA)
# ===========================
def augment_linear_meshes(mesh_list, num_gen=5, random_state=42):
    """
    Génère des meshes augmentés par PCA linéaire sur les coordonnées des vertices.

    Args:
        mesh_list (list[trimesh.Trimesh]): Liste de meshes originaux.
        num_gen (int): Nombre de meshes à générer par PCA.
        random_state (int): Seed pour reproductibilité.

    Returns:
        list[trimesh.Trimesh]: Meshes augmentés.
    """
    rng = np.random.default_rng(random_state)
    all_vertices = np.array([m.vertices.flatten() for m in mesh_list])
    pca = PCA(n_components=0.999, random_state=random_state)
    pca.fit(all_vertices)

    augmented = []
    for _ in range(num_gen):
        coeffs = rng.normal(scale=0.3, size=pca.n_components_)
        new_flat = pca.mean_ + coeffs @ pca.components_
        new_verts = new_flat.reshape(-1, 3)
        new_mesh = trimesh.Trimesh(vertices=new_verts, faces=mesh_list[0].faces, process=False)
        augmented.append(new_mesh)
    return augmented

# ===========================
# CONVERSION MESH -> TENSORS / MATRICES / GRAPH
# ===========================
def meshes_to_data(mesh_list: List[trimesh.Trimesh], mode: str = "flat") -> List:
    """
    Convertit une liste de meshes en tenseurs ou graphes PyG.

    Args:
        mesh_list (list[trimesh.Trimesh]): Liste de meshes à convertir.
        mode (str): 'flat', '3D' ou 'graph' pour différents formats de sortie.

    Returns:
        list[torch.Tensor | Data]: Liste de tenseurs ou objets Data PyG.
    """
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
    """
    Reconstruit des meshes à partir de tenseurs ou objets Data PyG.

    Args:
        data_list (list[torch.Tensor | Data]): Liste de données à convertir.
        mode (str): 'flat', '3D' ou 'graph'.
        faces (torch.Tensor): Faces pour reconstruire le mesh.

    Returns:
        list[trimesh.Trimesh]: Liste de meshes reconstruits.
    """
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
    """
    Crée des DataLoaders PyTorch ou PyG pour les splits train/val/test.

    Args:
        formatted_datasets (dict): Dictionnaire {'train': [...], 'val': [...], 'test': [...]} contenant les données.
        batch_size (int): Taille des batchs.
        shuffle (bool): Mélanger les données (uniquement pour train).
        mode (str): 'flat', '3D' ou 'graph' indiquant le type de données.
        num_workers (int): Nombre de workers pour DataLoader classique.

    Returns:
        dict: Dictionnaire de DataLoaders {'train': loader, 'val': loader, 'test': loader}.
    """
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
    """
    Reconstruit un tableau 3D de vertices à partir d'un tableau plat (flattened).

    Args:
        flat_vertices (np.ndarray): Tableau de shape (N, D) où D = num_vertices * dim.
        dim (int): Dimension des coordonnées (par défaut 3).

    Returns:
        np.ndarray: Tableau de shape (N, num_vertices, dim) avec les vertices reconstruits.
    """
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

    # -----------------------------------------
    # SPLIT
    # -----------------------------------------
    print("=== Création des splits ===")
    train_mesh, val_mesh, test_mesh = create_splits(meshes, split_ratios=(0.6,0.2,0.2))
    print(f"Train: {len(train_mesh)}  |  Val: {len(val_mesh)}  |  Test: {len(test_mesh)}\n")

    # -----------------------------------------
    # CLUSTER ANATOMIQUE GLOBAL
    # -----------------------------------------
    print("=== Détection des points anatomiques globaux ===")
    centroids = detect_anatomical_points(meshes, n_clusters=7, random_state=42)
    print("Centroids shape:", centroids.shape, "\n")

    # -----------------------------------------
    # LABELS DES MESHES
    # -----------------------------------------
    print("=== Application des labels aux meshes originaux ===")
    train_labels_list = [color_mesh_by_centroids(m, centroids)[1] for m in train_mesh]
    val_labels_list   = [color_mesh_by_centroids(m, centroids)[1] for m in val_mesh]
    test_labels_list  = [color_mesh_by_centroids(m, centroids)[1] for m in test_mesh]
    print("Labels OK.\n")

    # -----------------------------------------
    # AUGMENTATION LINÉAIRE (PCA)
    # -----------------------------------------
    print("=== Augmentation linéaire (PCA) train ===")
    train_aug_lin = augment_linear_meshes(train_mesh, num_gen=5, random_state=42)
    train_aug_lin_labels = [color_mesh_by_centroids(m, centroids)[1] for m in train_aug_lin]

    print("=== Augmentation linéaire (PCA) val ===")
    val_aug_lin = augment_linear_meshes(val_mesh, num_gen=3, random_state=42)
    val_aug_lin_labels = [color_mesh_by_centroids(m, centroids)[1] for m in val_aug_lin]

    # -----------------------------------------
    # CONFIGURATION DES RÉGIONS
    # -----------------------------------------
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
        "head": {"deforms":[
            {"type":"scaling","factor":1.02,"radius":0.20},
            {"type":"bump","intensity":0.03,"radius":0.20},
            {"type":"twist","axis_dir":[0,0,1],"max_angle":0.2,"radius":0.20}
        ]},
        "lesser_trochanter": {"deforms":[{"type":"scaling","factor":1.03,"radius":0.08}]},
        "greater_trochanter": {"deforms":[{"type":"scaling","factor":1.02,"radius":0.10}]},
        "body": {"deforms":[{"type":"scaling","factor":1.01,"radius":0.40}]},
        "popliteal_patellar_surface": {"deforms":[{"type":"scaling","factor":1.01,"radius":0.20}]},
        "medial_condyle": {"deforms":[{"type":"scaling","factor":1.03,"radius":0.10}]},
        "lateral_condyle": {"deforms":[{"type":"scaling","factor":1.03,"radius":0.10}]},
    }

    # -----------------------------------------
    # AUGMENTATION NON-LINÉAIRE
    # -----------------------------------------
    print("=== Augmentation non-linéaire sur meshes PCA (train) ===")
    train_aug_nl = augment_non_linear_meshes(
        train_aug_lin, train_aug_lin_labels,
        region_deformations, region_map,
        augmentation_factor=2, n_centers_per_region=1, device='cuda', random_state=42
    )
    print(f"→ {len(train_aug_nl)} meshes non-linéaires\n")

    print("=== Augmentation non-linéaire sur meshes PCA (val) ===")
    val_aug_nl = augment_non_linear_meshes(
        val_aug_lin, val_aug_lin_labels,
        region_deformations, region_map,
        augmentation_factor=1, n_centers_per_region=1, device='cuda', random_state=42
    )
    print(f"→ {len(val_aug_nl)} meshes non-linéaires\n")

    # -----------------------------------------
    # AUGMENTATION GÉOMÉTRIQUE SIMPLE
    # -----------------------------------------
    aug_config = {
        'rotation_prob': 0.7, 'rotation_max_angle': 15.0,
        'scale_prob': 0.5, 'scale_range': (0.95, 1.05),
        'jitter_prob': 0.5, 'jitter_std': 0.005,
        'noise_prob': 0.5, 'noise_std': 0.01,
        'elastic_prob': 0.3, 'elastic_alpha': 0.02, 'elastic_sigma': 0.05
    }

    train_aug_geo = augment_geometric_meshes(train_aug_lin, aug_config, device='cuda')
    val_aug_geo   = augment_geometric_meshes(val_aug_lin, aug_config, device='cuda')

    # -----------------------------------------
    # COMBINAISON DES AUGMENTATIONS
    # -----------------------------------------
    train_all = train_mesh + train_aug_lin + train_aug_nl + train_aug_geo
    val_all   = val_mesh   + val_aug_lin   + val_aug_nl   + val_aug_geo
    test_all  = test_mesh

    print(f"Total train meshes: {len(train_all)}")
    print(f"Total val meshes:   {len(val_all)}")
    print(f"Total test meshes:  {len(test_all)}\n")

    # -----------------------------------------
    # CONVERSION EN TENSEURS FLAT
    # -----------------------------------------
    X_train_flat = meshes_to_data(train_all, mode="flat")
    X_val_flat   = meshes_to_data(val_all, mode="flat")
    X_test_flat  = meshes_to_data(test_all, mode="flat")

    datasets_flat = {'train': X_train_flat, 'val': X_val_flat, 'test': X_test_flat}
    loaders_flat  = create_data_loaders(datasets_flat, batch_size=8, mode="flat")
    print("DataLoaders flat OK.\n")

    # -----------------------------------------
    # CONVERSION EN GRAPHES
    # -----------------------------------------
    X_train_graph = meshes_to_data(train_all, mode="graph")
    X_val_graph   = meshes_to_data(val_all, mode="graph")
    X_test_graph  = meshes_to_data(test_all, mode="graph")

    datasets_graph = {'train': X_train_graph, 'val': X_val_graph, 'test': X_test_graph}
    loaders_graph  = create_data_loaders(datasets_graph, batch_size=4, mode="graph")
    print("DataLoaders graph OK.\n")

    print("=== Pipeline terminé ===")
