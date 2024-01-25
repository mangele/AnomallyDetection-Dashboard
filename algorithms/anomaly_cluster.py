import sys
import umap
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
import shap


class AnomalyCluster:
    class SessionValues:
        def __init__(self, anomalies_batch, ind):
            self.timestamp = anomalies_batch["ticks"][ind]
            self.status_code = anomalies_batch["status_code"][ind]
            self.board = anomalies_batch["board"][ind]
            self.state = anomalies_batch["state"][ind]
            self.signal = anomalies_batch["signal"][ind]
            self.prev_value = anomalies_batch["prev_value"][ind]
            self.anomaly_value = anomalies_batch["anomaly_value"][ind]
            self.next_value = anomalies_batch["next_value"][ind]
            self.change_direction = anomalies_batch["change_direction"][ind]
            self.change_distance = anomalies_batch["change_distance"][ind]
            self.anomaly_score = anomalies_batch["anomaly_score"][ind]
            self.signal_importance = anomalies_batch["feature_importance"][ind]
            self.prev_value_std = anomalies_batch["prev_value_std"][ind]
            self.anomaly_value_std = anomalies_batch["anomaly_value_std"][ind]
            self.next_value_std = anomalies_batch["next_value_std"][ind]
            self.signal_change_direction = self.signal + "|change_direction"
            self.signal_change_distance = self.signal + "|change_distance"
            self.signal_anomaly_score = self.signal + "|anomaly_score"
            self.signal_anomaly_prev_value = self.signal + "|prev_value"
            self.signal_anomaly_value = self.signal + "|anomaly_value"
            self.signal_anomaly_next_value = self.signal + "|next_value"
            self.signal_importance_value = self.signal + "|importance"
            self.signal_prev_value_std = self.signal + "|prev_value_std"
            self.signal_anomaly_value_std = self.signal + "|anomaly_value_std"
            self.signal_next_value_std = self.signal + "|next_value_std"

    @staticmethod
    def get_category_map(categories):
        index = 0
        category_map = {}
        for code in np.unique(categories.cat.codes):
            category_map[code] = categories.cat.categories[index]
            index = index + 1
        return category_map

    @staticmethod
    def map_categorical_features(anomalies):
        board_categories = anomalies["board"].astype("category")
        state_categories = anomalies["state"].astype("category")

        board_map = __class__.get_category_map(board_categories)
        state_map = __class__.get_category_map(state_categories)

        anomalies.loc[:, "board"] = board_categories.cat.codes
        anomalies.loc[:, "state"] = state_categories.cat.codes

        return anomalies, board_map, state_map

    @staticmethod
    def add_signal_columns(df, session_values):
        signal_df = DataFrame([], 
                              columns=[session_values.signal_change_direction, 
                                       session_values.signal_change_distance,
                                       session_values.signal_anomaly_score,
                                       session_values.signal_anomaly_prev_value,
                                       session_values.signal_anomaly_value,
                                       session_values.signal_anomaly_next_value,
                                       session_values.signal_importance_value,
                                       session_values.signal_prev_value_std,
                                       session_values.signal_anomaly_value_std,
                                       session_values.signal_next_value_std
                                    ])
        df = pd.concat([df, signal_df], axis=1)
        return df

    @staticmethod
    def add_session(df, batch_name, session_values):
        new_row = {
            "fp": batch_name,
            "ticks": session_values.timestamp,
            "status_code": session_values.status_code,
            "board": session_values.board,
            "state": session_values.state,
            session_values.signal_change_direction: session_values.change_direction,
            session_values.signal_change_distance: session_values.change_distance,
            session_values.signal_anomaly_score: session_values.anomaly_score,
            session_values.signal_anomaly_prev_value: session_values.prev_value,
            session_values.signal_anomaly_value: session_values.anomaly_value,
            session_values.signal_anomaly_next_value: session_values.next_value,
            session_values.signal_importance_value: session_values.signal_importance,
            session_values.signal_prev_value_std: session_values.prev_value_std,
            session_values.signal_anomaly_value_std: session_values.anomaly_value_std,
            session_values.signal_next_value_std: session_values.next_value_std,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        return df

    @staticmethod
    def update_session(df, session_values, df_prev_index):
        if (
            pd.isnull(df[session_values.signal_change_direction][df_prev_index])
            or df[session_values.signal_change_direction][df_prev_index] == 0
            or session_values.signal_importance > df[session_values.signal_importance_value][df_prev_index]
            or session_values.anomaly_score > df[session_values.signal_anomaly_score][df_prev_index]
        ):
            df.loc[df_prev_index,session_values.signal_change_direction] = session_values.change_direction
            df.loc[df_prev_index,session_values.signal_change_distance] = session_values.change_distance
            df.loc[df_prev_index,session_values.signal_anomaly_score] = session_values.anomaly_score
            df.loc[df_prev_index,session_values.signal_anomaly_prev_value] = session_values.prev_value
            df.loc[df_prev_index,session_values.signal_anomaly_value] = session_values.anomaly_value
            df.loc[df_prev_index,session_values.signal_anomaly_next_value] = session_values.next_value
            df.loc[df_prev_index,session_values.signal_importance_value] = session_values.signal_importance
            df.loc[df_prev_index,session_values.signal_prev_value_std] = session_values.prev_value_std
            df.loc[df_prev_index,session_values.signal_anomaly_value_std] = session_values.anomaly_value_std
            df.loc[df_prev_index,session_values.signal_next_value_std] = session_values.next_value_std

    @staticmethod
    def generate_cluster_sessions(anomalies):
        df = DataFrame(
            {
                "fp": Series(dtype="str"),
                "ticks": Series(dtype="int64"),
                "status_code": Series(dtype="int16"),
                "board": Series(dtype="int8"),
                "state": Series(dtype="int8"),
            }
        )

        batch_names = anomalies["series_id"].unique()
        for batch_name in batch_names:
            anomalies_batch = anomalies[anomalies["series_id"] == batch_name].sort_values(by="ticks")

            df_prev_index = None
            for ind in anomalies_batch.index:
                session_values = __class__.SessionValues(anomalies_batch, ind)
                
                if session_values.signal_change_direction not in df.columns:
                    df = __class__.add_signal_columns(df, session_values)

                if df_prev_index is None:
                    df = __class__.add_session(df, batch_name, session_values)
                else:
                    __class__.update_session(df, session_values, df_prev_index)

                df_prev_index = len(df) - 1

        df.fillna(0, inplace=True)
        return df

    @staticmethod
    def remove_info_columns(anomalies):
        # define columns to keep
        cols = [
            col
            for col in anomalies.columns.values
            if col not in ["fp", "ticks", "status_code", "state", "board"]
        ]
        cols = [
            col
            for col in cols
            if (
               not col.endswith("|prev_value")
               and not col.endswith("|anomaly_value")
               and not col.endswith("|next_value")
            )
        ]
        return anomalies[anomalies.columns.intersection(cols)]

    @staticmethod
    def scale_dimensions(anomalies):
        anomalies_scaled = MinMaxScaler().fit(anomalies.values).transform(anomalies.values)
        return DataFrame(anomalies_scaled, columns=anomalies.columns.values)

    @staticmethod
    def reduce_dimensions(anomalies, dims):
        pca = PCA(n_components=0.95)
        pca_components = pca.fit_transform(anomalies.values)

        dims = min(int(pca.n_components_ / 2), len(anomalies.columns) - 1) 
        neighbors = min(dims - 1, len(anomalies) - 1)
        umap_model = umap.UMAP(n_components=dims,
                               n_neighbors=neighbors,
                               min_dist=0.0,
                               metric='correlation',
                               random_state=0)

        reduced_anomalies = umap_model.fit_transform(pca_components)
        return DataFrame(reduced_anomalies, columns=umap_model.get_feature_names_out())
        
    @staticmethod
    def cluster_kmeans(cluster_df):
        num_clusters = None
        K = range(2, 12, 1)
        prev_sil_coeff = 0
       
        for k in K:
            km = KMeans(n_clusters=k, random_state=0).fit(cluster_df)
            sil_coeff = silhouette_score(
                cluster_df.values, km.labels_, metric="euclidean"
            )
            if sil_coeff > prev_sil_coeff:
                prev_sil_coeff = sil_coeff
                num_clusters = k

        km = KMeans(n_clusters=num_clusters, random_state=0).fit(cluster_df.values)
        cluster_map = DataFrame()
        cluster_map["data_index"] = cluster_df.index.values
        cluster_map["cluster"] = km.labels_

        return cluster_map

    @staticmethod
    def cluster_gm(cluster_df):
        cluster_df_values = cluster_df.values
        num_clusters = None
        N = range(2, 12, 1)
        prev_bic = sys.float_info.max

        for n in N:
            gm = GaussianMixture(n_components=n, random_state=0).fit(cluster_df_values)
            bic = gm.bic(cluster_df_values)
            if prev_bic > bic:
                prev_bic = bic
                num_clusters = n

        gm = GaussianMixture(n_components=num_clusters, random_state=0)
        cluster_map = DataFrame()
        cluster_map["data_index"] = cluster_df.index.values
        cluster_map["cluster"] = gm.fit_predict(cluster_df_values)
        return cluster_map

    @staticmethod
    def cluster_hdbscan(cluster_df):
        cluster_df_values = cluster_df.values
        hdb = HDBSCAN(min_cluster_size=10)
        
        cluster_map = DataFrame()
        cluster_map["data_index"] = cluster_df.index.values
        cluster_map["cluster"] = hdb.fit_predict(cluster_df_values)
        return cluster_map

    @staticmethod
    def get_important_features(cluster_df, cluster_labels, limit = 5):
        clf = RandomForestClassifier().fit(cluster_df.values, cluster_labels)        
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(cluster_df.values)
        
        unique_cluster_labels = np.sort(np.unique(cluster_labels)).tolist()
        clusters_important_features = {}
        for ind in range(len(shap_values)):
            cluster_shap_values = shap_values[ind]
            shap_mean_values = np.mean(np.absolute(cluster_shap_values - np.mean(cluster_shap_values, axis=0)), axis=0)
            shap_mean_values_index_desc = np.argsort(-shap_mean_values)

            important_columns = []
            count = 0
            for index in shap_mean_values_index_desc:
                if count == limit:
                    break
                
                col = cluster_df.columns[index]
                if "|" in col:
                    base_col_name = col.split("|")[0]
                    if base_col_name in important_columns:
                        continue
                    col = base_col_name
                
                important_columns.append(col)
                count = count + 1

            clusters_important_features[unique_cluster_labels[ind]] = important_columns

        return clusters_important_features

    @staticmethod
    def is_feature_important(feature, important_features):
        if "|" in feature:
            base_feature_name = feature.split("|")[0]
            if base_feature_name not in important_features:
                return False
        return True

    @staticmethod
    def build_clusters_output(
        anomalies,
        board_map,
        state_map,
        cluster_map
    ):
        rows = []
        cols = anomalies.columns.values.tolist()
        cols.append("cluster")

        for cluster_label in np.unique(cluster_map["cluster"]):
            df = anomalies[
                anomalies.index.isin(
                    cluster_map[cluster_map.cluster == cluster_label]["data_index"]
                )
            ]
            for ind in df.index:
                row = []
                row.append(df["fp"][ind])
                row.append(float(df["ticks"][ind]))
                row.append(float(df["status_code"][ind]))
                row.append(board_map[df["board"][ind]])
                row.append(state_map[df["state"][ind]])

                for col in anomalies.columns.values:
                    if col in ["fp", "ticks", "status_code", "board", "state"]:
                        continue

                    row.append(float(df[col][ind]))

                row.append(float(cluster_label))
                rows.append(row)

        return rows, cols

    @staticmethod
    def build_cluster_output(
        cluster_sessions,
        board_map,
        state_map,
        cluster_map,
        cluster_label,
        cluster_important_features
    ):
        df = cluster_sessions[
            cluster_sessions.index.isin(
                cluster_map[cluster_map.cluster == cluster_label]["data_index"]
            )
        ]

        cols = []
        for feature in cluster_sessions.columns.values.tolist():
            # if not __class__.is_feature_important(feature, cluster_important_features):
            #     continue
            cols.append(feature)
        
        rows = []
        for ind in df.index:
            row = []
            row.append(df["fp"][ind])
            row.append(float(df["ticks"][ind]))
            row.append(float(df["status_code"][ind]))
            row.append(board_map[df["board"][ind]])
            row.append(state_map[df["state"][ind]])

            for col in cluster_sessions.columns.values.tolist():
                if col in ["fp", "ticks", "status_code", "board", "state"]:
                    continue
                # if not __class__.is_feature_important(col, cluster_important_features):
                    #continue
                row.append(float(df[col][ind]))

            rows.append(row)

        num_info_cols = 5 # fp, ticks, status_code, board, state
        np_rows = np.array(np.array(rows)[:,num_info_cols:], dtype=float)        
        cols_to_delete = []
        for col_index in range(num_info_cols, len(cols)):
            if np.all(np.isclose(np_rows[:, col_index - num_info_cols], 0)):
                cols_to_delete.append(col_index)
        
        for col_index in reversed(cols_to_delete):
            cols = np.delete(np.array(cols), col_index, 0).tolist()
            rows = np.delete(np.array(rows), col_index, 1).tolist()

        return rows, cols

