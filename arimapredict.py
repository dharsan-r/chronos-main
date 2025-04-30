import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def run_arima_prediction(test_name):
    """
    Run ARIMA predictions similar to the original Chronos pipeline approach
    
    Parameters:
    - test_name: Name of the numpy file to load
    """
    # Load data
    test_name = "./numpys/" + test_name
    mini_df = pd.DataFrame(np.load(test_name))
    
    # Prediction configuration
    p_length = 64  # Prediction length
    
    avg_RMSE, std_RMSE, avg_MASE, std_MASE, avg_WQL, std_WQL, worst, best, \
    truth_worst, truth_best, context_worst, context_best, forecast_index = test_df(mini_df, p_length)

    # File paths for saving plots
    file_name_best = test_name[:-4] + "best.png"
    file_name_worst = test_name[:-4] + "worst.png"
    
    # Print metrics
    print(test_name)
    print("STD RMSE: ", std_RMSE)
    print("Avg RMSE: ", avg_RMSE)
    print("Avg MASE: ", avg_MASE)
    print("STD MASE: ", std_MASE)
    print("Avg WQL: ", avg_WQL)
    print("STD WQL: ", std_WQL)
    print("Worst RMSE: ", mean_squared_error(worst, truth_worst))
    print("Best RMSE: ", mean_squared_error(best, truth_best))
    
    # Plot worst case
    plt.figure(1)
    plt.plot(context_worst, color="royalblue", label="historical data")
    plt.plot(forecast_index, truth_worst, color="gold", label="truth")
    plt.plot(forecast_index, worst, color="tomato", label="median forecast")
    plt.legend()
    plt.grid()
    plt.savefig(file_name_worst)
    plt.close()

    # Plot best case
    plt.figure(2)
    plt.plot(context_best, color="royalblue", label="historical data")
    plt.plot(forecast_index, truth_best, color="gold", label="truth")
    plt.plot(forecast_index, best, color="tomato", label="median forecast")
    plt.legend()
    plt.grid()
    plt.savefig(file_name_best)
    plt.close()

    print()

def test_df(df, p_length):
    """
    Perform ARIMA predictions and calculate metrics
    
    Parameters:
    - df: DataFrame with time series data
    - p_length: Prediction length
    
    Returns:
    - Various performance metrics and predictions
    """
    columns = df.columns
    train_cols, test_cols = train_test_split(columns, test_size=1, random_state=42)

    test_df = df[test_cols]
    num_predictions = test_df.shape[1]
    
    # Prepare data for prediction
    test_last_x = [test_df[col].tail(p_length).tolist() for col in test_df.columns]
    test_previous = [test_df[col].iloc[:-p_length].tolist() for col in test_df.columns]
    test_df = test_df.iloc[:-p_length].reset_index(drop=True)
    
    final = []
    RMSE = []
    MASE = []
    WQL = []
    min_index = max_index = 0
    min_value = 1000
    max_value = 0

    for x in range(num_predictions):
        # Prepare data for this specific column
        series = test_df.iloc[:, x]
        actual = test_last_x[x]
        
        # Fit ARIMA model
        try:
            model = ARIMA(series, order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=p_length)
        except Exception as e:
            print(f"ARIMA prediction failed for column {x}: {e}")
            forecast = [np.nan] * p_length
        
        final.append(forecast)
        
        # RMSE calculation
        mse = mean_squared_error(actual, forecast)
        rmse = np.sqrt(mse)
        
        if mse < min_value:
            min_index = x
            min_value = mse
        if mse > max_value:
            max_index = x
            max_value = mse
        
        RMSE.append(rmse)
        
        # MASE calculation
        previous_values = test_previous[x]
        if len(previous_values) > 1:
            naive_errors = np.abs(np.array(previous_values[1:]) - np.array(previous_values[:-1]))
            naive_mae = np.mean(naive_errors)
            
            model_errors = np.abs(np.array(actual) - np.array(forecast))
            model_mae = np.mean(model_errors)
            
            mase_value = model_mae / naive_mae if naive_mae > 0 else np.nan
            MASE.append(mase_value)
        else:
            MASE.append(np.nan)
        
        # WQL calculation (simplified version)
        actual_array = np.array(actual)
        forecast_array = np.array(forecast)
        wql = np.mean(np.abs(actual_array - forecast_array) / np.abs(actual_array))
        WQL.append(wql)

    # Calculate metrics
    avg_RMSE = sum(RMSE) / len(RMSE)
    std_RMSE = np.std(RMSE)
    
    avg_MASE = np.nanmean(MASE)
    std_MASE = np.nanstd(MASE)
    
    avg_WQL = np.mean(WQL)
    std_WQL = np.std(WQL)
    
    # Select best and worst predictions
    worst = final[max_index]
    best = final[min_index]
    truth_worst = test_last_x[max_index]
    truth_best = test_last_x[min_index]
    context_worst = test_df.iloc[:, max_index].tolist()
    context_best = test_df.iloc[:, min_index].tolist()
    
    forecast_index = range(len(context_best), len(context_best) + p_length)
    
    return (avg_RMSE, f"{std_RMSE:.10f}", avg_MASE, f"{std_MASE:.10f}", 
            avg_WQL, f"{std_WQL:.10f}", worst, best, truth_worst, 
            truth_best, context_worst, context_best, forecast_index)

# List of test names to process
test_names = [
    "test_control_Pabs02.npy", 
    "test_stroke_Pabs02.npy", 
    "test_control_Vabs02.npy", 
    "test_stroke_Vabs02.npy"
]

# Run predictions for each test name
for test_name in test_names:
    run_arima_prediction(test_name)