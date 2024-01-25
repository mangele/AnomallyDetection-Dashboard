import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import numpy as np

class DimensionalityReduction:
    def __init__(self, data):
        self.data = data
        self.imputed_data = self.imputer(data)

    def imputer(self, data):
		# Impute remaining NaN values
        data =  data.dropna(axis=1, how='all')
        imputer = KNNImputer(n_neighbors=5, weights="uniform")
        return imputer.fit_transform(data)

    def apply_pca(self, variance_threshold=0.95, show_plot=False):
        """
        Apply PCA to the data to reduce its dimensionality.

        Parameters:
        variance_threshold (float): The amount of variance to be preserved.
        show_plot (bool): Whether to show the explained variance plot.

        Returns:
        np.ndarray: Transformed data after applying PCA.
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.imputed_data)

        pca = PCA(n_components=variance_threshold)
        pca_components = pca.fit_transform(scaled_data)

        if show_plot:
            fig = plt.figure(figsize=(8, 5))
            #plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.title("Explained Variance by Components")
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            fig.tight_layout()
            plt.show()
            plt.savefig('pca_components.png')

        return pca_components

    def apply_umap(self, data=None, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
        """
        Apply UMAP to the data for non-linear dimensionality reduction.

        Parameters:
        n_components (int): The number of dimensions to reduce to.
        n_neighbors (int): The size of local neighborhood (in terms of number of neighboring sample points).
        min_dist (float): The effective minimum distance between embedded points.
        random_state (int): Random seed.

        Returns:
        np.ndarray: Transformed data after applying UMAP.
        """
        data_to_reduce = self.imputed_data if data is None else data

        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        umap_components = reducer.fit_transform(self.imputed_data)
        return umap_components



## Example usage:
## Assuming 'transformed_data' is your DataFrame after transformation
#reducer = DimensionalityReduction(transformed_data)
#pca_data = reducer.apply_pca(variance_threshold=0.95, show_plot=True)
#umap_data = reducer.apply_umap(n_components=3)

