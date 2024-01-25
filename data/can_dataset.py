class CanDataset:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def slice_data(self, start_tick, end_tick):
        if len(self.data) == 0:
            return self.data

        data = self.data[
            (self.data["ticks"] >= start_tick) & (self.data["ticks"] <= end_tick)
        ]

        return data

    def set_data(self, data):
        self.data = data


class CanDatasetInfo:
    def __init__(self, ds):
        self.ds_name = ds.name
        self.fp = ds.data.filename.iloc[0]
        self.sn = self.fp.split("/")[2]
        self.series_id = ds.data.series_id.iloc[0]
        self.min_tick = ds.data.ticks.iloc[0]
        self.max_tick = ds.data.ticks.iloc[-1]