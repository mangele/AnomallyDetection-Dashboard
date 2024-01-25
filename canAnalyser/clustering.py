from sklearn.cluster import HDBSCAN
from sklearn.mixture import GaussianMixture

class Clustering:
    def __init__(self, data):
        self.data = data

    def apply_hdbscan(self, min_cluster_size=5):
        """
        Apply HDBSCAN clustering to the data.

        Parameters:
        min_cluster_size (int): The minimum size of clusters.

        Returns:
        HDBSCAN: HDBSCAN clustering instance.
        """
        hdbscan_cluster = HDBSCAN(min_cluster_size=min_cluster_size)
        hdbscan_cluster.fit(self.data)
        return hdbscan_cluster

    def apply_gmm(self, n_components=6, random_state=0):
        """
        Apply Gaussian Mixture Modeling to the data.

        Parameters:
        n_components (int): The number of mixture components.
        random_state (int): Random state for reproducibility.

        Returns:
        GaussianMixture: GMM clustering instance.
        """
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(self.data)
        return gmm

## Example usage:
## Assuming 'reduced_data' is your data after dimensionality reduction
#clusterer = Clustering(reduced_data)
#
## Apply HDBSCAN
#hdbscan_clusters = clusterer.apply_hdbscan(min_cluster_size=5)
#
## Apply GMM
#gmm_clusters = clusterer.apply_gmm(n_components=6)
