from sympy import im
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import os
from sklearn.model_selection import train_test_split
import trimesh
import torch
import numpy as np

class MeshDataset(Dataset):
    def __init__(self, random_state=42):
        super().__init__()
        self.random_state = random_state
        self.dir = "./data"
        self.files = os.listdir(self.dir)
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.faces = None
    
    def mesh_to_edge_index(self, mesh):
        faces = mesh.faces
        
        edges = set()

        for (i, j, k) in faces:
            edges.add((i, j))
            edges.add((j, k))
            edges.add((k, i))
            edges.add((j, i))
            edges.add((k, j))
            edges.add((i, k))

        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        return edge_index
    
    def pre_process(self):
        return
    
    def load_data(self, dir="./data", validation=True, test_split_ratio=0.2, val_split_ratio=0.2, random_state=42):
        self.dir = dir
        self.files = os.listdir(self.dir)
        self.random_state = random_state
        meshes = []
        for file in self.files:
            if file.endswith('.obj'):
                mesh = trimesh.load(os.path.join(self.dir, file))
                vertices = torch.tensor(mesh.vertices, dtype=torch.float)
                edge_index = self.mesh_to_edge_index(mesh)
                data = Data(x=vertices, edge_index=edge_index)
                meshes.append(data)
        if meshes:
            self.faces = mesh.faces
        dataset_train, dataset_test = train_test_split(meshes, test_size=test_split_ratio, random_state=self.random_state)
        if validation:
            dataset_train, dataset_val = train_test_split(dataset_train, test_size=val_split_ratio, random_state=self.random_state)
            self.dataset_val = dataset_val
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
    
    def get_all_vertices(self, dataset):
        all_vertices = []
        for data in dataset:
            all_vertices.append(data.x.numpy())
        return np.array(all_vertices)
    
    def update_dataset_vertices(self, dataset, new_vertices):
        for i, data in enumerate(dataset):
            data.x = torch.tensor(new_vertices[i], dtype=torch.float)

    def reconstruct_mesh_from_dataset(self, dataset, index=0, use_faces=True):
        data = dataset[index]
        vertices = data.x.numpy()
        if use_faces and self.faces is not None:
            mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces)
        else:
            # Reconstruct from vertices and edges only
            #TODO: implement if needed
            return None
        return mesh

    def reconstruct_mesh_from_vertices(self, vertices, use_faces=True):
        if use_faces and self.faces is not None:
            mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces)
        else:
            # Reconstruct from vertices and edges only
            #TODO: implement if needed
            return None
        return mesh
    
    def flatten_vertices(self, vertices):
        flat_vertices = vertices.reshape((vertices.shape[0], -1))
        return flat_vertices

    def reconstruct_vertices_from_flat_vertices(self, flat_vertices, augmented_dim=0):
        N = flat_vertices.shape[0]
        D = 3 + augmented_dim
        M = flat_vertices.shape[1] // D
        vertices = flat_vertices.reshape((N, M, D))
        return vertices
    

    
    
