from pathlib import Path
from typing import List, Union
import pandas as pd
import numpy as np
from gluonts.dataset.arrow import ArrowWriter
import load_data2
from sklearn.model_selection import train_test_split

# for whole dataset it is 2304 sensor readings 9 * 256,    if not try 768
combined_Vabs_df, combined_Pabs_df, control_Pabs_data_df, control_Vabs_data_df, stroke_Pabs_data_df, stroke_Vabs_data_df, combined_headers = load_data2.return_data(768)

# Get the list of columns from the DataFrame
columns =combined_Vabs_df.columns

# change to 0.2 later
train_cols, test_cols = train_test_split(columns, test_size=0.2, random_state=42)

# Select the corresponding columns from the DataFrame
train_df = combined_Vabs_df[train_cols]
test_df = combined_Vabs_df[test_cols]

num_predictions = test_df.shape[1]
# change to 256 later
p_length = 256

np.save('test', test_df)

# removes the last 64 columns from test df for prediction
test_last_x = [test_df[col].tail(p_length).tolist() for col in test_df.columns]
test_df = test_df.iloc[:-p_length].reset_index(drop=True)

array_list = [train_df[col].to_numpy() for col in train_df.columns]

def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    compression: str = "lz4",
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and
        time_series.ndim == 2
    )

    # Set an arbitrary start time
    start = np.datetime64("2000-01-01 00:00", "s")

    dataset = [
        {"start": start, "target": ts} for ts in time_series
    ]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


if __name__ == "__main__":
    # Generate 20 random time series of length 1024

    # Convert to GluonTS arrow format
    convert_to_arrow("./train-data.arrow", time_series=array_list)