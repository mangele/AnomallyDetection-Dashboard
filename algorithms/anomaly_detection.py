from sklearn.ensemble import IsolationForest
from pandas import DataFrame as PandasDataFrame
from sklearn.preprocessing import LabelEncoder
import numpy as np
import shap
import time

class AnomalyDetection:
    @staticmethod
    def find_anomalies(state_boundary, ds):
        fault_data = ds.slice_data(
            state_boundary["start_tick"], state_boundary["end_tick"]
        )

        if len(fault_data) == 0:
            return PandasDataFrame()

        return __class__.detect_anomalies(fault_data)

    @staticmethod
    def detect_anomalies(fault_data):
        fault_data = __class__.prepare_data(fault_data)

        if len(fault_data) == 0 or len(fault_data.columns) == 1:
            return PandasDataFrame()

        clf_start_time = time.time()
        clf = IsolationForest(n_estimators=100, random_state=0, contamination=0.1).fit(
            fault_data.values
        )

        print(f"IsolationForest fitting time: {time.time() - clf_start_time} seconds")

        prediction_start_time = time.time()
        anomalies = clf.predict(fault_data.values)
        print(f"Prediction time: {time.time() - prediction_start_time} seconds")

        scores = clf.decision_function(fault_data.values)

        scores = [-1 * s + 0.5 for s in scores]

        fault_data = __class__.filter_important_features(clf, fault_data)
        fault_data = __class__.filter_important_samples(
            anomalies, scores, fault_data
        )

        return fault_data

    @staticmethod
    def prepare_data(data):
        # Replace missing values with 0 for numeric columns
        numeric_cols = data.select_dtypes(include=np.number).columns
        data[numeric_cols] = data[numeric_cols].fillna(0)

        # Identify categorical columns (including those with 'object' type)
        categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns

        # Label encode categorical columns
        for col in categorical_cols:
            # Initialize LabelEncoder
            le = LabelEncoder()

            # Fill missing values with a placeholder string and convert to string type
            data[col] = data[col].fillna('Missing').astype(str)

            # Apply LabelEncoder
            data[col] = le.fit_transform(data[col])

        # Remove rows where all values are zero
        data = data[~(data == 0).all(axis=1)]
        # Process each column in the DataFrame
        for col in data.columns:
            # Skip the 'ticks' column
            if col == "ticks":
                continue

            # Apply np.isclose only on original numeric columns
            if col in numeric_cols:
                # Drop the column if all its values are close to the first value
                if np.all(np.isclose(data[col].values, data[col].values[0])):
                    data.drop(col, axis=1, inplace=True)
                    continue

            # Drop the column based on specific name conditions
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ["serialnumber", "deviceid", "flags", "sp", "target", "limit", "power"]):
                data.drop(col, axis=1, inplace=True)
        print(f"------{data.dtypes}-------")
        return data

    @staticmethod
    def filter_important_features(clf, fault_data):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(fault_data.values)

        np_shap_values = np.array(shap_values)
        for col_index in range(np_shap_values.shape[1]):
            col = fault_data.columns[col_index]
            if col == "ticks":
                continue
            fault_data[col + "|importance"] = np_shap_values[:, col_index]

        return fault_data

    @staticmethod
    def filter_important_samples(anomalies, scores, fault_data):
        fault_data_indices = fault_data.index
        anomaly_indices = []
        anomaly_scores = []

        for index, result in enumerate(anomalies):
            if result < 0:
                anomaly_indices.append(fault_data_indices[index])
                anomaly_scores.append(scores[index])

        fault_data = fault_data[fault_data.index.isin(anomaly_indices)]
        fault_data["anomaly_score"] = anomaly_scores
        return fault_data

    @staticmethod
    def prepare_signal_data(signal, anomalies, fault_data):
        anomalies = anomalies[anomalies[signal] != 0]
        fault_data = fault_data[fault_data[signal] != 0]

        if len(anomalies) == 0 or len(fault_data) == 0:
            return fault_data, anomalies

        anomalies = __class__.detect_signal_outliers(
            fault_data, anomalies, signal
        )
        return fault_data, anomalies

    @staticmethod
    def detect_signal_outliers(fault_data, anomalies, col):
        col_values = fault_data[col].values
        if col_values.dtype == bool:
            col_values = col_values.astype(int)
        q25 = np.percentile(col_values, 25)
        q75 = np.percentile(col_values, 75)
        iqr = q75 - q25
        cut_off = iqr * 1.5
        lower = q25 - cut_off
        upper = q75 + cut_off
        anomalies = anomalies[(anomalies[col] < lower) | (anomalies[col] > upper)]
        return anomalies
