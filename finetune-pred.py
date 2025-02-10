import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import load_data2
from sklearn.model_selection import train_test_split

from chronos import ChronosPipeline

def run_finetune(pipe_name, test_name):
    
    name = "./scripts/training/output/" + pipe_name + "/checkpoint-final"
    
    pipeline = ChronosPipeline.from_pretrained(
        name,
        device_map="cuda:0",  # use "cpu" for CPU inference and "mps" for Apple Silicon
        torch_dtype=torch.float64,
    )
    
    mini_df = pd.DataFrame(np.load(test_name))
    
    avg_RMSE, worst, best, truth_worst, truth_best, context_worst, context_best, forecast_index = test_df(mini_df, pipeline)

    file_name_best = test_name[:-4] + "best.png"
    file_name_worst = test_name[:-4] + "worst.png"
    a = test_name[-4]
    print(a," Avg_RMSE:",avg_RMSE)
    print(a," Worst_RMSE:",mean_squared_error(best, truth_best))
    print(a," Best_RMSE:", mean_squared_error(worst, truth_worst))
    
    
    plt.figure(1)
    plt.plot(context_worst, color="royalblue", label="historical data")
    plt.plot(forecast_index, truth_worst, color="gold", label="truth")
    plt.plot(forecast_index, worst, color="tomato", label="median forecast")
    plt.legend()
    plt.grid()
    plt.savefig(file_name_best)
    plt.close()

    plt.figure(2)
    plt.plot(context_best, color="royalblue", label="historical data")
    plt.plot(forecast_index, truth_best, color="gold", label="truth")
    plt.plot(forecast_index, best, color="tomato", label="median forecast")
    plt.legend()
    plt.grid()
    plt.savefig(file_name_worst)
    plt.close()

    print()


def test_df(df, pipeline):
    columns = df.columns
    train_cols, test_cols = train_test_split(columns, test_size=1, random_state=42)

    test_df = df[test_cols]
    num_predictions = test_df.shape[1]
    p_length = 256
    
    test_last_x = [test_df[col].tail(p_length).tolist() for col in test_df.columns]
    test_df = test_df.iloc[:-p_length].reset_index(drop=True)
    
    forecasts = []
    for col in test_df.columns:
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
    
    
    forecast_index = range(len(context_best), len(context_best) + p_length)
    
    return avg_RMSE, worst, best, truth_worst, truth_best, context_worst, context_best, forecast_index
    
    
pipe_name = "run-0"
test_name = "test_Pabs03.npy"

pipe_names = ["run-0","run-1","run-2","run-3","run-4", "run-5", "run-6", "run-7", "run-8", "run-9"]
test_names = ["test_Pabs08.npy", "test_Vabs08.npy", "test_Pabs07.npy", "test_Vabs07.npy","test_Pabs05.npy", "test_Vabs05.npy","test_Pabs03.npy", "test_Vabs03.npy","test_Pabs02.npy", "test_Vabs02.npy"]

for x in range(len(pipe_names)):
    run_finetune(pipe_names[x], test_names[x])