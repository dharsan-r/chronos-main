from pathlib import Path
from typing import List, Union
import pandas as pd
import numpy as np
from gluonts.dataset.arrow import ArrowWriter
import load_data2
from sklearn.model_selection import train_test_split

# for whole dataset it is 2304 sensor readings 9 * 256,    if not try 768
combined_Vabs_df, combined_Pabs_df, control_Pabs_data_df, control_Vabs_data_df, stroke_Pabs_data_df, stroke_Vabs_data_df, combined_headers = load_data2.return_data(576)



def make_arrow_npy(size):
    # Get the list of columns from the DataFrame
    columns = combined_Vabs_df.columns
    
    # First do a train-test split on the entire dataset to maintain overall ratio
    train_cols, test_cols = train_test_split(columns, test_size=size, random_state=42)
    
    # Now separate the test columns into control and stroke
    test_control_cols = [col for col in test_cols if list(columns).index(col) < 599]  # columns 0-598
    test_stroke_cols = [col for col in test_cols if list(columns).index(col) >= 599]   # remaining columns
    
    # Select the corresponding columns from the DataFrame for training
    train_df_Vabs = combined_Vabs_df[train_cols]
    test_df_Vabs = combined_Vabs_df[test_cols]
    
    train_df_Pabs = combined_Pabs_df[train_cols]
    test_df_Pabs = combined_Pabs_df[test_cols]
    
    # Create separate test dataframes for control and stroke
    test_control_df_Vabs = combined_Vabs_df[test_control_cols]
    test_stroke_df_Vabs = combined_Vabs_df[test_stroke_cols]
    
    test_control_df_Pabs = combined_Pabs_df[test_control_cols]
    test_stroke_df_Pabs = combined_Pabs_df[test_stroke_cols]
    
    # Set up file names
    test_name_vabs = 'test_Vabs0' + str(round(size*10))
    test_control_name_vabs = 'test_control_Vabs0' + str(round(size*10))
    test_stroke_name_vabs = 'test_stroke_Vabs0' + str(round(size*10))
    
    train_name_vabs = './train_Vabs0' + str(round(round(1-size,1)*10)) + ".arrow"
    
    test_name_pabs = 'test_Pabs0' + str(round(size*10))
    test_control_name_pabs = 'test_control_Pabs0' + str(round(size*10))
    test_stroke_name_pabs = 'test_stroke_Pabs0' + str(round(size*10))
    
    train_name_pabs = './train_Pabs0' + str(round(round(1-size,1)*10)) + ".arrow"
    
    # Save all test datasets
    np.save(test_name_vabs, test_df_Vabs)
    np.save(test_control_name_vabs, test_control_df_Vabs)
    np.save(test_stroke_name_vabs, test_stroke_df_Vabs)
    
    np.save(test_name_pabs, test_df_Pabs)
    np.save(test_control_name_pabs, test_control_df_Pabs)
    np.save(test_stroke_name_pabs, test_stroke_df_Pabs)
    
    # Prepare training data for arrow format
    array_list_Vabs = [train_df_Vabs[col].to_numpy() for col in train_df_Vabs.columns]
    array_list_Pabs = [train_df_Pabs[col].to_numpy() for col in train_df_Pabs.columns]
    
    # Convert to arrow format
    convert_to_arrow(train_name_vabs, time_series=array_list_Vabs)
    convert_to_arrow(train_name_pabs, time_series=array_list_Pabs)

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
    # test_sizes = [0.2, 0.3, 0.5, 0.7, 0.8]
    test_sizes = [0.2]

    for sizes in test_sizes:
        make_arrow_npy(sizes)