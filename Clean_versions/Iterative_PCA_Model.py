import random
from re import X
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Iterative_PCA():
    def __init__(self, n_components=5, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.tol = tol
        self.pca = PCA(n_components=n_components, random_state=random_state)      
        self.train_mean = None

    def fit_filled(self, X_filled):
        self.train_mean = np.mean(X_filled, axis=0)
        X_centered = X_filled - self.train_mean
        self.pca.fit(X_centered)
    
    def transform(self, X):
        X_centered = X - self.train_mean
        return self.pca.transform(X_centered)
    
    def inverse_transform(self, X_transformed):
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        return X_reconstructed + self.train_mean
    
    def fit_iterative_unknown(self, X_partial):
        mask = np.isnan(X_partial)
        X_filled = X_partial.copy()
        self.train_mean = np.nanmean(X_partial, axis=0)
        X_filled[mask] = np.broadcast_to(self.train_mean, X_partial.shape)[mask]

        for i in range(self.max_iter):
            self.fit_filled(X_filled)
            X_reconstructed = self.inverse_transform(self.transform(X_filled))
            diff = np.linalg.norm((X_reconstructed - X_filled)[mask]) / np.sum(mask)

            X_filled[mask] = X_reconstructed[mask]

            if diff < self.tol:
                print(f"Converged at iteration {i+1}")
                break
        
        return X_filled
    
    def fit_iterative_known(self, X_train_partial, X_train_original, X_val_partial=None, X_val_original=None, max_iter=50, Early_Stopping=False, patience=5):
        train_mask = np.isnan(X_train_partial)
        if X_val_partial is not None:
            val_mask = np.isnan(X_val_partial)

        self.train_mean = np.nanmean(X_train_partial, axis=0)
        X_train_filled = X_train_partial.copy()
        X_train_filled[train_mask] = np.broadcast_to(self.train_mean, X_train_partial.shape)[train_mask]

        if X_val_partial is not None:
            X_val_filled = X_val_partial.copy()
            X_val_filled[val_mask] = np.broadcast_to(self.train_mean, X_val_partial.shape)[val_mask]

        best_val_miss_error = float('inf')
        patience_counter = 0

        train_errors_obs, val_errors_obs = [], []
        train_errors_miss, val_errors_miss = [], []
        global_errors_train_, global_errors_val_ = [], []

        for i in range(max_iter):
            self.fit_filled(X_train_filled)
            X_train_reconstructed = self.inverse_transform(self.transform(X_train_filled))
            diff_train = np.linalg.norm((X_train_reconstructed - X_train_filled)[train_mask]) / np.sum(train_mask)

            X_train_filled[train_mask] = X_train_reconstructed[train_mask]

            train_error_obs, train_error_miss, global_error_train = self.compute_error(X_train_reconstructed, X_train_original, mask=train_mask)

            train_errors_obs.append(train_error_obs)
            train_errors_miss.append(train_error_miss)
            global_errors_train_.append(global_error_train)

            if X_val_partial is not None:
                X_val_reconstructed = self.inverse_transform(self.transform(X_val_filled))
                val_error_obs, val_error_miss, global_error_val = self.compute_error(X_val_reconstructed, X_val_original, mask=val_mask)

                val_errors_obs.append(val_error_obs)
                val_errors_miss.append(val_error_miss)
                global_errors_val_.append(global_error_val)

                X_val_filled[val_mask] = X_val_reconstructed[val_mask]
                current_val_miss_error = val_error_miss
            else:
                current_val_miss_error = train_error_miss
            
            if Early_Stopping:
                if current_val_miss_error < best_val_miss_error:
                    best_val_miss_error = current_val_miss_error
                    best_U = self.pca.components_.T
                    best_S = self.pca.singular_values_
                    best_Vt = self.pca.components_
                    best_X_train = X_train_filled.copy()
                    if X_val_partial is not None:
                        best_X_val = X_val_filled.copy()
                    iteration_best = i
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping at iteration:", i + 1)
                        break
            
        if Early_Stopping:
            print("Best iteration:", iteration_best + 1)
            self.pca.components_ = best_Vt
            self.pca.singular_values_ = best_S
            X_train_filled = best_X_train
            if X_val_partial is not None:
                X_val_filled = best_X_val
        
        errors = {'train_obs': train_errors_obs,
                  'val_obs': val_errors_obs,
                  'train_miss': train_errors_miss,
                  'val_miss': val_errors_miss,
                  'global_train': global_errors_train_,
                  'global_val': global_errors_val_}

        return X_train_filled, X_val_filled if X_val_partial is not None else None, errors
    
    def inference(self, X_partial):
        mask = np.isnan(X_partial)
        X_filled = X_partial.copy()
        X_filled[mask] = np.broadcast_to(self.train_mean, X_partial.shape)[mask]

        X_reconstructed = self.inverse_transform(self.transform(X_filled))
        X_filled[mask] = X_reconstructed[mask]

        return X_filled
    
    def plot_explained_variance(self):
        plt.figure(figsize=(8, 5))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_)*100)
        plt.xlabel('Nombre de composantes principales')
        plt.ylabel('Variance expliquée cumulée (%)')
        plt.title('Variance expliquée par les composantes principales')
        plt.grid()
        plt.show()
    
    def plot_errors(self, obs_error, miss_error, global_error, title_suffix=''):
        plt.figure(figsize=(12, 5))
        plt.suptitle(f'Erreurs itératives {title_suffix}', fontsize=16)

        plt.subplot(1, 3, 1)
        plt.plot(obs_error, marker='o')
        plt.title('Erreur sur les données observées')
        plt.xlabel('Itérations')
        plt.ylabel('Erreur')
        plt.grid()

        plt.subplot(1, 3, 2)
        plt.plot(miss_error, marker='o', color='orange')
        plt.title('Erreur sur les données manquantes')
        plt.xlabel('Itérations')
        plt.ylabel('Erreur')
        plt.grid()

        plt.subplot(1, 3, 3)
        plt.plot(global_error, marker='o', color='green')
        plt.title('Erreur totale')
        plt.xlabel('Itérations')
        plt.ylabel('Erreur')
        plt.grid()
        plt.tight_layout()
        plt.show()
    
    def plot_train_errors(self, errors):
        self.plot_errors(errors['train_obs'], errors['train_miss'], errors['global_train'], title_suffix='(Train)')
        self.plot_errors(errors['val_obs'], errors['val_miss'], errors['global_val'], title_suffix='(Validation)')
    
    def compute_error(self, X_reconstructed, X_original, mask=None):
        if mask is None:
            mask = np.ones_like(X_original, dtype=bool)
            return np.linalg.norm(X_reconstructed - X_original)/(X_original.shape[0]*X_original.shape[1])
        err_obs = np.linalg.norm(X_reconstructed[~mask] - X_original[~mask])/sum(sum(~mask))
        err_miss = np.linalg.norm(X_reconstructed[mask] - X_original[mask])/sum(sum(mask))
        global_err = np.linalg.norm(X_reconstructed - X_original)/(X_original.shape[0]*X_original.shape[1])
        return err_obs, err_miss, global_err

if __name__ == "__main__":
    # Example usage
    import numpy as np
    import matplotlib.pyplot as plt
    from Iterative_PCA_Model import Iterative_PCA
    from sklearn.model_selection import train_test_split
    import dataloader as dl

    random_state = 42
    np.random.seed(random_state)

    # Load data
    # Must be common for all models
    meshes, files = dl.load_femur_meshes('./data')

    vertices = [dl.mesh_to_vector(mesh) for mesh in meshes]
    vertices = np.array(vertices)

    X_train, X_test = train_test_split(vertices, test_size=0.2, random_state=random_state)
    X_train, X_val, = train_test_split(X_train, test_size=0.25, random_state=random_state)  # 0.25 x 0.8 = 0.2

    X_train_partial = X_train.copy()
    X_val_partial = X_val.copy()
    X_test_partial = X_test.copy()

    # Without missing values
    print("Fitting Iterative PCA without missing values...")
    iterative_pca = Iterative_PCA(n_components=0.98, tol=1e-4)
    iterative_pca.fit_filled(X_train)

    X_test_reconstructed = iterative_pca.inverse_transform(iterative_pca.transform(X_test))
    X_test_error = iterative_pca.compute_error(X_test_reconstructed, X_test)
    
    #Worst test error
    print(f"Test Error without missing values: {X_test_error}")


    # Introduce missing values
    print("Introducing missing values...")
    missing_rate = 0.1
    n_missing_train = int(np.floor(missing_rate * X_train_partial.size))
    n_missing_val = int(np.floor(missing_rate * X_val_partial.size))
    n_missing_test = int(np.floor(missing_rate * X_test_partial.size))

    missing_indices_train = (np.random.randint(0, X_train_partial.shape[0], n_missing_train),
                             np.random.randint(0, X_train_partial.shape[1], n_missing_train))
    missing_indices_val = (np.random.randint(0, X_val_partial.shape[0], n_missing_val),
                           np.random.randint(0, X_val_partial.shape[1], n_missing_val))
    missing_indices_test = (np.random.randint(0, X_test_partial.shape[0], n_missing_test),
                            np.random.randint(0, X_test_partial.shape[1], n_missing_test))
    
    X_train_partial[missing_indices_train] = np.nan
    X_val_partial[missing_indices_val] = np.nan
    X_test_partial[missing_indices_test] = np.nan

    # Fit Iterative PCA
    iterative_pca = Iterative_PCA(n_components=0.98, tol=1e-4)
    X_train_filled, X_val_filled, errors = iterative_pca.fit_iterative_known(
        X_train_partial, X_train, X_val_partial, X_val,
        max_iter=100, Early_Stopping=True, patience=20
    )

    # Plot explained variance
    iterative_pca.plot_explained_variance()

    # Plot training and validation errors
    iterative_pca.plot_train_errors(errors)

    # Inference on test set
    X_test_filled = iterative_pca.inference(X_test_partial)
    test_error_obs, test_error_miss, test_global_error = iterative_pca.compute_error(
        X_test_filled, X_test, mask=np.isnan(X_test_partial)
    )
    print(f"Test Error on Observed Data: {test_error_obs}")
    print(f"Test Error on Missing Data: {test_error_miss}")
    print(f"Test Global Error: {test_global_error}")