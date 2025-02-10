import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import load_data2
from sklearn.model_selection import train_test_split

from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    "./scripts/training/output/run-20train/checkpoint-final",
    device_map="cuda:0",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.float64,
)

# pipeline = ChronosPipeline.from_pretrained(
#     "amazon/chronos-t5-small",
#     device_map="cuda:0",  # use "cpu" for CPU inference and "mps" for Apple Silicon
#     torch_dtype=torch.float64,
# )

# for whole dataset it is 2304 sensor readings 9 * 256,    if not try 768
combined_Vabs_df, combined_Pabs_df, control_Pabs_data_df, control_Vabs_data_df, stroke_Pabs_data_df, stroke_Vabs_data_df, combined_headers = load_data2.return_data(2304)

def test_df(df, sizes):
    # Get the list of columns from the DataFrame
    columns = df.columns

    # change to 0.2 later
    train_cols, test_cols = train_test_split(columns, test_size=sizes, random_state=42)

    # Select the corresponding columns from the DataFrame
    train_df = df[train_cols]
    test_df = df[test_cols]

    num_predictions = test_df.shape[1]
    # change to 256 later
    p_length = 256

    test_last_x = [test_df[col].tail(p_length).tolist() for col in test_df.columns]
    test_df = test_df.iloc[:-p_length].reset_index(drop=True)

    forecasts = []
    for col in tqdm(test_df.columns, desc="Processing columns"):
        tensor = torch.tensor(test_df[col].values)
        
        forecasts.append(
            pipeline.predict(
                context=tensor,
                num_samples=20,
                prediction_length=p_length,
                limit_prediction_length=False
        ))
        
    final =[]
    RMSE = []
    min_index=max_index=0
    min_vlaue = 1000
    max_value = 0
    
    for x in range(num_predictions):
        median = np.quantile(forecasts[x][0].numpy(), 0.5 , axis=0)
        final.append(median)
        actual = test_last_x[x]
        mse = mean_squared_error(actual, median) # type: ignore
        
        if mse<min_vlaue:
            min_index = x
            min_vlaue = mse
        if mse>max_value:
            max_index = x
            max_value = mse
        
        RMSE.append(mse)

    avg_RMSE = sum(RMSE) / len(RMSE)

    worst = final[max_index]
    best = final[min_index]
    truth_worst = test_last_x[max_index]
    truth_best = test_last_x[min_index]
    context_worst = test_df.iloc[:, max_index].tolist()
    context_best = test_df.iloc[:, min_index].tolist()
    
    print('worst: ',  mean_squared_error(worst, truth_worst) )
    print('best: ',  mean_squared_error(best, truth_best) )
    
    forecast_index = range(len(context_best), len(context_best) + p_length)
    
    
    print(avg_RMSE)
    
    return avg_RMSE, worst, best, truth_worst, truth_best, context_worst, context_best, forecast_index


test_sizes = [0.2, 0.3, 0.5, 0.7, 0.8]

for sizes in test_sizes:
    avg_RMSE, worst, best, truth_worst, truth_best, context_worst, context_best, forecast_index = test_df(combined_Vabs_df, sizes)
    name1 = "vabs_worst" + str(sizes) + ".png"
    name2 = "vabs_best" + str(sizes) + ".png"



    plt.figure(1)
    plt.plot(context_worst, color="royalblue", label="historical data")
    plt.plot(forecast_index, truth_worst, color="gold", label="truth")
    plt.plot(forecast_index, worst, color="tomato", label="median forecast")
    plt.legend()
    plt.grid()
    plt.savefig(name1)
    plt.close()

    plt.figure(2)
    plt.plot(context_best, color="royalblue", label="historical data")
    plt.plot(forecast_index, truth_best, color="gold", label="truth")
    plt.plot(forecast_index, best, color="tomato", label="median forecast")
    plt.legend()
    plt.grid()
    plt.savefig(name2)
    plt.close()

    
    avg_RMSE, worst, best, truth_worst, truth_best, context_worst, context_best, forecast_index = test_df(combined_Pabs_df, sizes)
    name1 = "pabs_worst" + str(sizes) + ".png"
    name2 = "pabs_best" + str(sizes) + ".png"
        
    plt.figure(1)
    plt.plot(context_worst, color="royalblue", label="historical data")
    plt.plot(forecast_index, truth_worst, color="gold", label="truth")
    plt.plot(forecast_index, worst, color="tomato", label="median forecast")
    plt.legend()
    plt.grid()
    plt.savefig(name1)
    plt.close()

    plt.figure(2)
    plt.plot(context_best, color="royalblue", label="historical data")
    plt.plot(forecast_index, truth_best, color="gold", label="truth")
    plt.plot(forecast_index, best, color="tomato", label="median forecast")
    plt.legend()
    plt.grid()
    plt.savefig(name2)
    plt.close()