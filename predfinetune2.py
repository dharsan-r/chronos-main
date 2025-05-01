import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from scipy import stats

from sklearn.model_selection import train_test_split

from chronos import ChronosPipeline

def run_finetune(pipe_name, test_name):
    
    name = "./scripts/training/output/" + pipe_name + "/checkpoint-final"
    print(test_name)
    test_name = "./numpys/" + test_name
    pred_name = test_name[:-4] +"_preds.npy"
    print(pred_name)
    
    pipeline = ChronosPipeline.from_pretrained(
        name,
        device_map="cuda:0",  # use "cpu" for CPU inference and "mps" for Apple Silicon
        torch_dtype=torch.float64,
    )
    
    mini_df = pd.DataFrame(np.load(test_name))
    
    avg_RMSE, std_RMSE, avg_MASE, std_MASE, avg_WQL, std_WQL, worst, best, truth_worst, truth_best, context_worst, context_best, forecast_index, final = test_df(mini_df, pipeline)

    print("Final")
    print(len(final))
    np.save(pred_name, final)
    
    
    
    file_name_best = test_name[:-4] + "best.png"
    file_name_worst = test_name[:-4] + "worst.png"
    print(test_name)
    print("STD RMSE: ", std_RMSE)
    print("Avg RMSE: ", avg_RMSE)
    print("Avg MASE: ", avg_MASE)
    print("STD MASE: ", std_MASE)
    print("Avg WQL: ", avg_WQL)
    print("STD WQL: ", std_WQL)
    print("Worst RMSE: ", mean_squared_error(worst, truth_worst))
    print("Best RMSE: ", mean_squared_error(best, truth_best))
    
    plt.figure(1)
    plt.plot(context_worst, color="royalblue", label="historical data")
    plt.plot(forecast_index, truth_worst, color="gold", label="truth")
    plt.plot(forecast_index, worst, color="tomato", label="median forecast")
    plt.legend()
    plt.grid()
    plt.savefig(file_name_worst)
    plt.close()

    plt.figure(2)
    plt.plot(context_best, color="royalblue", label="historical data")
    plt.plot(forecast_index, truth_best, color="gold", label="truth")
    plt.plot(forecast_index, best, color="tomato", label="median forecast")
    plt.legend()
    plt.grid()
    plt.savefig(file_name_best)
    plt.close()

    print()

def test_df(df, pipeline):
    # columns = df.columns
    # train_cols, test_cols = train_test_split(columns, test_size=1, random_state=42)
    
    # test_df = df[test_cols]
    # num_predictions = test_df.shape[1]
    p_length = 64
    
    columns = df.columns
    train_cols, test_cols = train_test_split(columns, test_size=1, random_state=42)

    test_df = df
    num_predictions = test_df.shape[1]
    
    # Prepare data for prediction
    test_last_x = [test_df[col].tail(p_length).tolist() for col in test_df.columns]
    test_previous = [test_df[col].iloc[:-p_length].tolist() for col in test_df.columns]
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
        
    final = []
    RMSE = []
    MASE = []
    WQL = []
    min_index = max_index = 0
    min_value = 1000
    max_value = 0
    
    quantile_levels = [0.1, 0.5, 0.9]  # Quantiles for WQL calculation
    
    for x in range(num_predictions):
        # Get the low, median, and high predictions directly
        low, median, high = np.quantile(forecasts[x][0].numpy(), [0.1, 0.5, 0.9], axis=0)
        final.append(median)
        actual = test_last_x[x]
        
        # RMSE calculation
        mse = mean_squared_error(actual, median)
        rmse = np.sqrt(mse)
        
        if mse < min_value:
            min_index = x
            min_value = mse
        if mse > max_value:
            max_index = x
            max_value = mse
        
        RMSE.append(rmse)
        
        # MASE calculation
        # Calculate naive forecast errors (using the previous value as prediction)
        previous_values = test_previous[x]
        if len(previous_values) > 1:
            naive_errors = np.abs(np.array(previous_values[1:]) - np.array(previous_values[:-1]))
            naive_mae = np.mean(naive_errors)
            
            # Calculate model errors
            model_errors = np.abs(np.array(actual) - np.array(median))
            model_mae = np.mean(model_errors)
            
            # MASE calculation (avoiding division by zero)
            if naive_mae > 0:
                mase_value = model_mae / naive_mae
            else:
                mase_value = np.nan
                
            MASE.append(mase_value)
        else:
            MASE.append(np.nan)
            
        # WQL calculation using the pre-calculated quantiles
        actual_array = np.array(actual)
        wql_values = []
        
        # For low quantile (q=0.1)
        errors_low = actual_array - low
        penalty_low = (errors_low >= 0).astype(float) * 0.9 + (errors_low < 0).astype(float) * 0.1
        quantile_loss_low = 2 * np.sum(np.abs(errors_low) * penalty_low) / np.sum(np.abs(actual_array))
        wql_values.append(quantile_loss_low)
        
        # For median quantile (q=0.5)
        errors_median = actual_array - median
        penalty_median = (errors_median >= 0).astype(float) * 0.5 + (errors_median < 0).astype(float) * 0.5
        quantile_loss_median = 2 * np.sum(np.abs(errors_median) * penalty_median) / np.sum(np.abs(actual_array))
        wql_values.append(quantile_loss_median)
        
        # For high quantile (q=0.9)
        errors_high = actual_array - high
        penalty_high = (errors_high >= 0).astype(float) * 0.1 + (errors_high < 0).astype(float) * 0.9
        quantile_loss_high = 2 * np.sum(np.abs(errors_high) * penalty_high) / np.sum(np.abs(actual_array))
        wql_values.append(quantile_loss_high)
        
        # Average the WQL across all quantiles
        WQL.append(np.mean(wql_values))

    avg_RMSE = sum(RMSE) / len(RMSE)
    std_RMSE = np.std(RMSE)  # Standard deviation of the RMSE values
    std_RMSE_precise = f"{std_RMSE:.10f}"
    
    # Calculate average and std for new metrics
    avg_MASE = np.nanmean(MASE)  # Using nanmean to handle potential NaN values
    std_MASE = np.nanstd(MASE)
    std_MASE_precise = f"{std_MASE:.10f}"
    
    avg_WQL = np.mean(WQL)
    std_WQL = np.std(WQL)
    std_WQL_precise = f"{std_WQL:.10f}"
    
    worst = final[max_index]
    best = final[min_index]
    truth_worst = test_last_x[max_index]
    truth_best = test_last_x[min_index]
    context_worst = test_df.iloc[:, max_index].tolist()
    context_best = test_df.iloc[:, min_index].tolist()
    
    forecast_index = range(len(context_best), len(context_best) + p_length)
    
    return (avg_RMSE, std_RMSE_precise, avg_MASE, std_MASE_precise, 
            avg_WQL, std_WQL_precise, worst, best, truth_worst, 
            truth_best, context_worst, context_best, forecast_index, final)
    
pipe_name = "run-1"
test_name = "test_control_Vabs08.npy"

# pipe_names = ["run-9","run-10","run-11","run-12","run-13", "run-14", "run-15", "run-16", "run-17", "run-18"]
# test_names = ["test_Pabs08.npy", "test_Vabs08.npy", "test_Pabs07.npy", "test_Vabs07.npy","test_Pabs05.npy", "test_Vabs05.npy","test_Pabs03.npy", "test_Vabs03.npy","test_Pabs02.npy", "test_Vabs02.npy"]

pipe_names = ["run-8", "run-8"]
test_names = ["test_control_Vabs02.npy", "test_stroke_Vabs02.npy"]


# run_finetune(pipe_name, test_name)

for x in range(len(pipe_names)):
    run_finetune(pipe_names[x], test_names[x])