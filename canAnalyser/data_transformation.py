import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataTransformation:
    def __init__(self, data):
        self.data = data

    def transform(self):
        """
        Transform categorical data into numerical.

        Returns:
        DataFrame: The transformed data.
        """
        try:
            # One hot encoding for specific columns if they exist
            if 'rt_ccu_chargingstate' in self.data.columns:
                self.data = pd.get_dummies(self.data, columns=['rt_ccu_chargingstate'])
            elif 'rt-ccu_chargingstate' in self.data.columns:
                self.data = pd.get_dummies(self.data, columns=['rt-ccu_chargingstate'])
        except Exception as e:
            print(f"Error during one-hot encoding: {e}")

        categorical_columns = self.data.select_dtypes(include=['object', 'string', 'bool']).columns

        # Initialize LabelEncoder
        label_encoders = {}
        for col in categorical_columns:
            if col == "filename":
                continue
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            label_encoders[col] = le  # Store the label encoder for each column

    def create_feature_vector(self, aggregation_method="mean"):
        """
        Create feature vectors by aggregating data.
    
        Parameters:
        aggregation_method (str): The method to use for aggregation ('mean', 'median', 'kurtosis', or 'all').
    
        Returns:
        DataFrame: The aggregated feature vectors.
        """
        if aggregation_method not in ["mean", "median", "kurtosis", "all"]:
            raise ValueError("Aggregation method must be either 'mean', 'median', 'kurtosis', or 'all'")
    
        if aggregation_method == "all":
            # Calculate mean, median, and kurtosis
            mean_data = self.data.groupby('filename').mean()
            median_data = self.data.groupby('filename').median()
            kurtosis_data = self.data.groupby('filename').apply(lambda x: x.kurt())
    
            # Concatenate horizontally
            aggregated_data = pd.concat([mean_data, median_data, kurtosis_data], axis=1)
    
            # Rename columns to reflect the metric
            mean_cols = [f"{col}_mean" for col in mean_data.columns]
            median_cols = [f"{col}_median" for col in median_data.columns]
            kurtosis_cols = [f"{col}_kurtosis" for col in kurtosis_data.columns]
            aggregated_data.columns = mean_cols + median_cols + kurtosis_cols
        else:
            # For mean, median, and kurtosis, use the existing approach
            if aggregation_method == "kurtosis":
                aggregated_data = self.data.groupby('filename').apply(lambda x: x.kurt())
            else:
                aggregation_function = getattr(self.data.groupby('filename'), aggregation_method)
                aggregated_data = aggregation_function()
    
        return aggregated_data
#        return self.data
    def create_feature_vector2(self, aggregation_method="mean"):
        """
        Create feature vectors by aggregating data.
    
        Parameters:
        aggregation_method (str): The method to use for aggregation ('mean', 'median', or 'kurtosis').
    
        Returns:
        DataFrame: The aggregated feature vectors.
        """
        if aggregation_method not in ["mean", "median", "kurtosis"]:
            raise ValueError("Aggregation method must be either 'mean', 'median', or 'kurtosis'")
    
        if aggregation_method == "kurtosis":
            # Use apply method to calculate kurtosis for each group
            aggregated_data = self.data.groupby('filename').apply(lambda x: x.kurt())
        else:
            # For mean and median, use the existing approach
            aggregation_function = getattr(self.data.groupby('filename'), aggregation_method)
            aggregated_data = aggregation_function()
    
        return aggregated_data



## Example usage:
## Assuming 'data' is your DataFrame after preprocessing
#transformer = DataTransformation(data)
#transformed_data = transformer.transform()
#
## Choose 'mean' or 'median' for aggregation
#feature_vector = transformer.create_feature_vector(aggregation_method="median")
