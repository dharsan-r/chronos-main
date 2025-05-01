import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import statistics

test_names = [
    "test_control_Vabs02", 
    "test_stroke_Vabs02"
]

test_name = test_names[0]

df = pd.read_csv('numpys/lag/control_Pabs_test.csv')
df = df.iloc[:, 1:]
print(df)
df = pd.DataFrame(np.load("./numpys/" + test_name +".npy"))
print(df)