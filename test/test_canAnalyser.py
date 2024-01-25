import json
import sys
import os
sys.path.append(os.path.abspath('..'))
import argparse
from canAnalyser.data_preprocessing import DataPreprocessing
from canAnalyser.data_transformation import DataTransformation
from canAnalyser.feature_vector_merger import FeatureVectorMerger
from canAnalyser.dimensionality_reduction import DimensionalityReduction
from canAnalyser.clustering import Clustering
from canAnalyser.feature_importance import FeatureImportance
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_settings(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def plot_umap(umap_df, error_codes, title="", prefix=""):
    # Unique error codes and color mapping
    unique_error_codes = np.unique(error_codes)
    cmap = plt.cm.tab10
    norm = plt.Normalize(vmin=0, vmax=len(unique_error_codes)-1)
    error_code_to_color = {code: cmap(norm(i)) for i, code in enumerate(unique_error_codes)}

    # Count occurrences of each error code
    counts = Counter(error_codes)

    # Plotting the results
    fig, ax = plt.subplots(figsize=(8, 5))

    for code in unique_error_codes:
        indices = np.where(error_codes == code)[0]
        cluster_data = umap_df.iloc[indices]
        ax.scatter(cluster_data['UMAP1'], cluster_data['UMAP2'], 
                  c=[error_code_to_color[code]]*len(cluster_data), s=50, alpha=0.4, edgecolors='black')
        # Calculate centroid of the cluster
        centroid = cluster_data.mean()
        ax.annotate(str(counts[code]), (centroid['UMAP1'], centroid['UMAP2']), 
                    textcoords="offset points", xytext=(0,10), ha='center')

    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title(f'CCU-ISO-WSGT-TS-FS UMAP Projection - {title} labels')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Create legend
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(label),
                                markersize=8, markerfacecolor=error_code_to_color[label]) for label in unique_error_codes]
    plt.legend(handles=legend_handles, title='Cluster', fontsize=8, edgecolor='k')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{prefix}_projection.png")

def create_combined_dataframe(feature_importance_dict, feature_vector, cluster_labels):
    # Create an empty dataframe to store the combined results
    combined_df = pd.DataFrame(columns=['session_id', 'cluster', 'Feature', 'Importance'])

    # Populate the combined dataframe with the data from each cluster
    for cluster, df in feature_importance_dict.items():
        for session_id in feature_vector.index[cluster_labels == cluster]:
            temp_df = df.copy()
            temp_df['session_id'] = session_id
            temp_df['cluster'] = cluster
            combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

    return combined_df


def main(settings_path):
    # Load settings from the specified JSON file
    settings = load_settings(settings_path)
    aggregation_method  = settings['AGGREGATION_METHOD']

    method = settings["FEATURE_IMPORTANCE_METHOD"]
    if method not in ["permutation", "cross-validation"]:
        raise ValueError(f"FeatureImportace method must be either permutation or cross-validation, but got {method}") 

    feature_vectors = {}
    for dataset_name, config in settings['DATA_CONFIG'].items():
        print(f"**********{dataset_name}**********")
        preprocessor = DataPreprocessing(year=settings['YEAR_TO_FILTER'])
        data = preprocessor.load_data(config['path'])
        data = preprocessor.drop_columns(data, config['drop_columns'])

        #
        transformer = DataTransformation(data)
        transformer.transform()
        feature_vector = transformer.create_feature_vector(aggregation_method)
        feature_vectors[dataset_name] = feature_vector

    # Feature Vector Merging
    merger = FeatureVectorMerger()
    merged_feature_vector, error_codes = merger.merge_feature_vectors(*feature_vectors.values())
    #print(merged_feature_vector)

    # Dimensionality Reduction
    reducer = DimensionalityReduction(merged_feature_vector)
    pca_data = reducer.apply_pca(variance_threshold=0.95, show_plot=True)
    umap_visualization = reducer.apply_umap(n_components=3, data=pca_data)
    umap_data = pd.DataFrame(data=umap_visualization, columns=['UMAP1', 'UMAP2', 'UMAP3'])
    umap_cluster = reducer.apply_umap(n_components=20, data=pca_data)

    # Clustering
    clusterer = Clustering(umap_cluster)
    hdbscan_clusters = clusterer.apply_hdbscan(min_cluster_size=5)
    gmm = clusterer.apply_gmm(n_components=7)

    gmm_clusters_label = gmm.predict(umap_cluster)
    hdb_clusters_label = hdbscan_clusters.labels_

    # Plotting cluster projection
    plot_umap(umap_data, error_codes, "Error Code Label", prefix=f"{aggregation_method}_error")
    plot_umap(umap_data, gmm_clusters_label, "GMM Cluster Label", prefix=f"{aggregation_method}_gmm")
    plot_umap(umap_data, hdb_clusters_label, "HDBSCAN Cluster Label", prefix=f"{aggregation_method}_hdb")

    # Classification and Feature Importance HDBSCAN
    feature_imp = FeatureImportance(merged_feature_vector, hdb_clusters_label)
    importances = feature_imp.calculate_importances_kfolds()
    feature_imp.plot_importances(importances, max_features=100, prefix=f"{aggregation_method}_hdb")

    fi = feature_imp.calculate_importances_per_cluster(method=method, prefix=f"{aggregation_method}_hdb_per_cluster")
    ready_output = create_combined_dataframe(fi, merged_feature_vector, hdb_clusters_label)
    ready_output.to_csv(f"{method}_{aggregation_method}_file.csv")

    # Classification and Feature Importance GMM
    #feature_imp = FeatureImportance(merged_feature_vector, gmm_clusters_label)
    #importances = feature_imp.calculate_importances_kfolds()
    #feature_imp.plot_importances(importances, max_features=100, prefix=f"{aggregation_method}_gmm")
    #feature_imp.calculate_importances_per_cluster(prefix=f"{aggregation_method}_gmm_per_cluster")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and analyze data")
    parser.add_argument('settings_file', type=str, help='Path to the JSON settings file')
    args = parser.parse_args()
    main(args.settings_file)
