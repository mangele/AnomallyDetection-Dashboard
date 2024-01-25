import pandas as pd

class DataPreprocessing:
    def __init__(self, year):
        self.year = year

    def load_data(self, filepath):
        """
        Load data from a CSV file.

        Parameters:
        filepath (str): The file path to the CSV file.

        Returns:
        DataFrame: The loaded data.
        """
        data = pd.read_csv(filepath)
        return self.filter_year(data)

    def filter_year(self, data):
        """
        Filter the data for the specified year.

        Parameters:
        data (DataFrame): The data to filter.

        Returns:
        DataFrame: The filtered data.
        """
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        data = data.dropna(subset=['timestamp'])
        return data[data['timestamp'].dt.year == self.year]

    def drop_columns(self, data, dropping_list):
        """
        Drop specified columns from the data.

        Parameters:
        data (DataFrame): The data from which to drop columns.
        dropping_list (list): A list of column names to drop.

        Returns:
        DataFrame: The data with specified columns dropped.
        """
        return data.drop(dropping_list, axis=1)


# Example usage:
#preprocessor = DataPreprocessing(year=2020)
#data = preprocessor.load_data('/path/to/your/csvfile.csv')
#dropping_list = ["column1", "column2", "column3"]
#data = preprocessor.drop_columns(data, dropping_list)

# Now 'data' is ready to be passed to the next stages of your pipeline.

