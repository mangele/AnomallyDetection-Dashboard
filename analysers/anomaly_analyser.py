from typing import Dict
import math
import numpy as np
from pandas import DataFrame as PandasDataFrame
from da.data.can_dataset import CanDataset, CanDatasetInfo
from da.algorithms.anomaly_detection import AnomalyDetection
import pandas as pd

class AnomalyAnalyser:
    @staticmethod
    def load_dataset(name, data):
        ds = CanDataset(name, data)
        return ds

    @staticmethod
    def find_state_boundaries(ccu_ds, state_column, error_column):
        post_fault_data = None
        try:
            post_fault_data = ccu_ds.data[
                (ccu_ds.data[error_column] > 0)
            ].sort_values(by="ticks")
        except:
            print("Error column not found in", ccu_ds.name, ccu_ds.data.columns.values.tolist())
            exit(0)

        if len(post_fault_data.index) == 0:
            return None

        fault_timestamp = post_fault_data.iloc[0]["ticks"]
        status_code = post_fault_data.iloc[0][error_column]

        pre_fault_data = ccu_ds.data[ccu_ds.data["ticks"] < fault_timestamp]
        current_state = pre_fault_data.iloc[-1][state_column]
        state_boundaries = []
        state_fault_data = pre_fault_data[
            pre_fault_data[state_column] == current_state
        ]
        
        start_tick = state_fault_data.iloc[0]["ticks"]
        end_tick = state_fault_data.iloc[-1]["ticks"]
        state_boundaries = {
            "state": current_state,
            "start_tick": start_tick,
            "end_tick": end_tick,
            "status_code": status_code,
        }

        return state_boundaries

    @staticmethod
    def process_board_anomalies(state_boundary, ds):
        anomalies = AnomalyDetection.find_anomalies(state_boundary, ds)
        if len(anomalies) == 0:
            return PandasDataFrame()
        return anomalies

    @staticmethod
    def plot_anomalies(ds, anomalies, state_boundary, directory="charts"):
        fault_data = ds.slice_data(
            state_boundary["start_tick"], state_boundary["end_tick"]
        )

        for col in fault_data.columns.tolist():
            if col not in anomalies.columns:
                continue

            if np.all(np.isclose(fault_data[col].values, fault_data[col].values[0])):
                continue

            col_fault_data, col_anomalies = AnomalyDetection.prepare_signal_data(
                col, anomalies, fault_data
            )

            if len(col_anomalies) == 0 or len(col_fault_data) == 0:
                continue

            colours = []
            for index in range(len(col_fault_data)):
                if not col_anomalies[
                    col_anomalies["ticks"] == col_fault_data.iloc[index]["ticks"]
                ].empty:
                    colours.append("red")
                else:
                    colours.append("black")

    @staticmethod
    def is_change_magnitude_important(prev, current, stdv_col_value):
        cutoff_diff = 0.5
        diff = abs(prev - current)
        is_important = diff >= cutoff_diff and stdv_col_value > 1
        return is_important

    @staticmethod
    def get_standarised_value(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)

    @staticmethod
    def get_change_boundary_distance(
        prev_val, next_val, min_val, max_val, delta_timestamp=2
    ):
        # delta_timestamp (delta_x) is two as default due to the log resolution.
        # Ticks are 1 tick apart from each other so
        # 2 = 1 tick from prev to anomaly value + 1 tock from anomaly to next value

        # standarise values
        prev_val = __class__.get_standarised_value(prev_val, min_val, max_val)
        next_val = __class__.get_standarised_value(next_val, min_val, max_val)

        # euclidean distance between the signal value before the anomaly and after the anomaly
        return math.sqrt(
            math.pow(delta_timestamp, 2) + math.pow(next_val - prev_val, 2)
        )

    @staticmethod
    def build_anomalies_output_row(index,
                                   col,
                                   anomalies,
                                   state_data,
                                   fp,
                                   series_id,
                                   sn,
                                   board,
                                   state,
                                   status_code,
                                   ccu_ds_info,
                                   min_col_value,
                                   max_col_value,
                                   stdv_col_value):

        anomaly_values = anomalies[
            anomalies["ticks"]
            == state_data.iloc[index]["ticks"]
        ]
        if anomaly_values.empty:
            return []

        anomaly_prev_value = 0
        anomaly_next_value = 0
        anomaly_value = anomaly_values.iloc[0][col]
        anomaly_timestamp = anomaly_values.iloc[0]["ticks"]
        playback_percentage = (
            (anomaly_timestamp-ccu_ds_info.min_tick) 
            / 
            (ccu_ds_info.max_tick-ccu_ds_info.min_tick)
        )
        anomaly_score = anomaly_values.iloc[0]["anomaly_score"]
        feature_importance = anomaly_values.iloc[0][col + "|importance"]
        
        if index > 0:
            anomaly_prev_value = state_data.iloc[index - 1][col]

        if index < (len(state_data) - 1):
            anomaly_next_value = state_data.iloc[index + 1][col]

        anomaly_prev_value_std = __class__.get_standarised_value(anomaly_prev_value,
                                                                 min_col_value,
                                                                 max_col_value)
        anomaly_value_std = __class__.get_standarised_value(anomaly_value,
                                                            min_col_value,
                                                            max_col_value)
        anomaly_next_value_std = __class__.get_standarised_value(anomaly_next_value,
                                                                 min_col_value,
                                                                 max_col_value)
        
        # if the change in magnitude is small, discard anomaly
        if not __class__.is_change_magnitude_important(anomaly_prev_value_std,
                                                       anomaly_value_std,
                                                       stdv_col_value):
            return []
        
        if anomaly_prev_value > -1 and anomaly_prev_value < 1:
            return []

        if anomaly_prev_value == 0:
            change_direction = 0
        elif anomaly_value > anomaly_prev_value:
            change_direction = 1
        elif anomaly_value < anomaly_prev_value:
            change_direction = -1
        else:
            change_direction = 0

        if change_direction == 0:
            return []
        
        change_distance = __class__.get_change_boundary_distance(
            anomaly_prev_value,
            anomaly_next_value,
            min_col_value,
            max_col_value,
        )
        
        return [
            str(fp),
            str(series_id),
            sn,
            board,
            state,
            col,
            float(anomaly_prev_value),
            float(anomaly_value),
            float(anomaly_next_value),
            change_direction,
            float(change_distance),
            float(anomaly_timestamp),
            float(status_code),
            float(anomaly_score),
            float(feature_importance),
            float(playback_percentage),
            float(anomaly_prev_value_std),
            float(anomaly_value_std),
            float(anomaly_next_value_std)
        ]

    @staticmethod
    def build_anomalies_output(boards_anomalies, ccu_ds_info):
        rows = []
        cols = ["fp","series_id","sn","board","state","signal",
                "prev_value","anomaly_value","next_value",
                "change_direction","change_distance","ticks",
                "status_code","anomaly_score","feature_importance", 
                "playback_percentage", "prev_value_std",
                "anomaly_value_std","next_value_std"]
        for board, result in boards_anomalies.items():
            state = result["state"]
            board_ds = result["ds"]
            state_boundary = result["state_boundary"]
            anomalies = result["anomalies"]
            fp = result["fp"]
            sn = result["sn"]
            series_id = result["series_id"]
            status_code = state_boundary["status_code"]
            state_data = board_ds.slice_data(
                state_boundary["start_tick"], state_boundary["end_tick"]
            )
           



            for col in state_data.columns.tolist():
                if col not in anomalies.columns:
                    continue

            # Check if the column is numeric before applying np.isclose
            if pd.api.types.is_numeric_dtype(state_data[col]):
                if np.all(np.isclose(state_data[col].values, state_data[col].values[0])):
                    continue
            else:
                # Handle or skip non-numeric data
                print(f"Skipping non-numeric column: {col}")
                continue


                if np.all(np.isclose(state_data[col].values, state_data[col].values[0])):
                    continue

                (
                    col_state_data,
                    col_anomalies,
                ) = AnomalyDetection.prepare_signal_data(col, anomalies, state_data)

                if len(col_anomalies) == 0 or len(col_state_data) == 0:
                    continue

                min_col_value = col_state_data[col].min()
                max_col_value = col_state_data[col].max()
                stdv_col_value = col_state_data[col].std()
                
                for index in range(len(col_state_data)):
                    row = __class__.build_anomalies_output_row(index,
                                                               col,
                                                               col_anomalies,
                                                               col_state_data,
                                                               fp,
                                                               series_id,
                                                               sn,
                                                               board,
                                                               state,
                                                               status_code,
                                                               ccu_ds_info,
                                                               min_col_value,
                                                               max_col_value,
                                                               stdv_col_value)
                    if not row:
                        continue
                    rows.append(row)
        return rows, cols

    @staticmethod
    def add_board_anomalies(boards_anomalies, state_boundaries, ds, ds_info):
        anomalies = __class__.process_board_anomalies(state_boundaries, ds)
        boards_anomalies[ds.name] = {
                "fp": ds_info.fp,
                "sn": ds_info.sn,
                "series_id": ds_info.series_id,
                "ds": ds,
                "state": state_boundaries["state"],
                "anomalies": anomalies,
                "state_boundary": state_boundaries,
        }

    @staticmethod
    def run(
        rt_datasets: Dict[str, PandasDataFrame],
        ccu_dataset_name: str,
        state_column: str,
        error_column: str,
    ):
        ccu_ds = __class__.load_dataset(
            ccu_dataset_name, rt_datasets[ccu_dataset_name]
        )
        ccu_ds_info = CanDatasetInfo(ccu_ds)
        state_boundaries = __class__.find_state_boundaries(ccu_ds, state_column, error_column)

        if (state_boundaries is None) or len(state_boundaries) == 0:
            return

        boards_anomalies = {}
        __class__.add_board_anomalies(boards_anomalies,
                                      state_boundaries,
                                      ccu_ds,
                                      ccu_ds_info)
        
        for ds_name, pd_ds in rt_datasets.items():
            if ds_name == ccu_dataset_name:
                continue
        
            ds = __class__.load_dataset(
                ds_name, pd_ds
            )

            if len(ds.data) == 0:
                continue

            ds_info = CanDatasetInfo(ds)
            __class__.add_board_anomalies(boards_anomalies,
                                          state_boundaries,
                                          ds,
                                          ds_info)
        return __class__.build_anomalies_output(boards_anomalies, ccu_ds_info)
