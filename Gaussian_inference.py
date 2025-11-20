import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

class Gaussian_Inference():
    def __init__(self):
        self.train_mean = None
        self.U = None
        self.S = None
        self.Vt = None
        self.n_components = None
    
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
     
    def fit_filled(self, X_filled, n_components=5):
        self.train_mean = np.mean(X_filled, axis=0)
        #We can't use covariance matrix because of high dimensionality
        U, S, Vt = svd(X_filled - self.train_mean, full_matrices=False)
        self.U = U[:, :n_components]
        self.S = S[:n_components]
        self.Vt = Vt[:n_components, :]
        self.n_components = n_components
    
    def fit_train_val_masked(self, X_train_partial, X_train_original, X_val_partial, X_val_original, n_components=5, max_iteration=100, Early_Stopping=True, max_patience=10, verbose=False):
        train_mask = np.isnan(X_train_partial)
        val_mask = np.isnan(X_val_partial)
        
        train_errors_obs = []
        val_errors_obs = []

        train_errors_miss = []
        val_errors_miss = []

        global_errors_train_ = []
        global_errors_val_ = []

        self.train_mean = np.nanmean(X_train_partial, axis=0)
        X_train_filled = X_train_partial.copy()
        X_train_filled[train_mask] = np.broadcast_to(self.train_mean, X_train_partial.shape)[train_mask]

        U, S, Vt = svd(X_train_filled - self.train_mean, full_matrices=False)
        self.U = U[:, :n_components]
        self.S = S[:n_components]
        self.Vt = Vt[:n_components, :]
        self.n_components = n_components
    
        err_train_obs, err_train_miss, global_train_err = self.compute_error(X_train_filled, X_train_original, train_mask)
        train_errors_obs.append(err_train_obs)
        train_errors_miss.append(err_train_miss)
        global_errors_train_.append(global_train_err)

        X_val_filled = self.inference(X_val_partial)

        err_val_obs, err_val_miss, global_val_err = self.compute_error(X_val_filled, X_val_original, val_mask)

        val_errors_obs.append(err_val_obs)
        val_errors_miss.append(err_val_miss)
        global_errors_val_.append(global_val_err)
        min_val_miss_error = val_errors_miss[-1]
        min_train_miss_error = train_errors_miss[-1]
        best_U = self.U
        best_S = self.S
        best_Vt = self.Vt
        best_X_train = X_train_filled.copy()
        best_X_val = X_val_filled.copy()
        iteration_best = 0
        patience=1

        for i in range(1, max_iteration):

            X_train_filled = self.inference(X_train_partial)
            X_val_filled = self.inference(X_val_partial)

            self.train_mean = np.mean(X_train_filled, axis=0)

            U, S, Vt = svd(X_train_filled - self.train_mean, full_matrices=False)
            self.U = U[:, :n_components]
            self.S = S[:n_components]
            self.Vt = Vt[:n_components, :]
                
            err_train_obs, err_train_miss, global_train_err = self.compute_error(X_train_filled, X_train_original, train_mask)
            train_errors_obs.append(err_train_obs)
            train_errors_miss.append(err_train_miss)
            global_errors_train_.append(global_train_err)

            err_val_obs, err_val_miss, global_val_err = self.compute_error(X_val_filled, X_val_original, val_mask)
            val_errors_obs.append(err_val_obs)
            val_errors_miss.append(err_val_miss)
            global_errors_val_.append(global_val_err)

            if verbose and i % 10 == 0:
                print(f"Iteration {i}, Train error on observed: {err_train_obs}")
                print(f"Iteration {i}, Train error on missing: {err_train_miss}")
                print(f"Iteration {i}, Train total error: {global_train_err}")
                print(f"Iteration {i}, Val error on observed: {err_val_obs}")
                print(f"Iteration {i}, Val error on missing: {err_val_miss}")
                print(f"Iteration {i}, Val total error: {global_val_err}")

            if Early_Stopping:
                if err_val_miss < min_val_miss_error and err_train_miss < min_train_miss_error:
                    min_val_miss_error = err_val_miss
                    min_train_miss_error = err_train_miss
                    patience = 1
                    best_U = self.U
                    best_S = self.S
                    best_Vt = self.Vt
                    iteration_best = i
                    best_X_train = X_train_filled.copy()
                    best_X_val = X_val_filled.copy()
                else:
                    patience += 1 
                if patience > max_patience:
                    print("Early stopping at iteration ", i)
                    print("Restoring best SVD from iteration ", iteration_best)
                    self.U = best_U
                    self.S = best_S
                    self.Vt = best_Vt
                    X_train_filled = best_X_train
                    X_val_filled = best_X_val
                    break
                
        if Early_Stopping:
            print("Best iteration:", iteration_best + 1)
            self.U = best_U
            self.S = best_S
            self.Vt = best_Vt
            X_train_filled = best_X_train
            X_val_filled = best_X_val
        
        errors = {'train_obs': train_errors_obs,
                  'val_obs': val_errors_obs,
                  'train_miss': train_errors_miss,
                  'val_miss': val_errors_miss,
                  'global_train': global_errors_train_,
                  'global_val': global_errors_val_}
        
        return X_train_filled, X_val_filled, errors
    
    def fit_inference(self, X_train_partial, max_iteration=100, tol=1e-4, n_components=5):
        train_mask = np.isnan(X_train_partial)
        
        self.train_mean = np.nanmean(X_train_partial, axis=0)
        X_train_filled = X_train_partial.copy()
        X_train_filled[train_mask] = np.broadcast_to(self.train_mean, X_train_partial.shape)[train_mask]

        U, S, Vt = svd(X_train_filled - self.train_mean, full_matrices=False)
        self.U = U[:, :n_components]
        self.S = S[:n_components]
        self.Vt = Vt[:n_components, :]
        self.n_components = n_components

        for i in range(1, max_iteration):
            X_train_new = self.inference(X_train_partial)

            diff = np.linalg.norm(X_train_new - X_train_filled)/(X_train_filled.shape[0]*X_train_filled.shape[1])
            X_train_filled = X_train_new

            self.train_mean = np.mean(X_train_filled, axis=0)

            U, S, Vt = svd(X_train_filled - self.train_mean, full_matrices=False)
            self.U = U[:, :self.n_components]
            self.S = S[:self.n_components]
            self.Vt = Vt[:self.n_components, :]

            if diff < tol:
                print("Converged at iteration ", i)
                break
        
        return X_train_filled

    def inference(self, X_partial):
        mask = np.isnan(X_partial)
        X_filled = X_partial.copy()
        for i in range(X_partial.shape[0]):

            ind_missing  = np.where(mask[i] == 1)[0]
            ind_observed = np.where(mask[i] == 0)[0]

            mu_miss = self.train_mean[ind_missing]
            mu_obs  = self.train_mean[ind_observed]

            V_obs  = self.Vt[:, ind_observed].T
            V_miss = self.Vt[:, ind_missing].T

            x_obs = (X_partial[i, ind_observed] - mu_obs)
            z = np.linalg.lstsq(V_obs, x_obs, rcond=None)[0]
            delta = V_miss @ z
            X_filled[i, ind_missing] = mu_miss + delta

        return X_filled
