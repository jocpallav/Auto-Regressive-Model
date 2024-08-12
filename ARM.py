import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

# Assuming observed_data is your observed time series data
np.random.seed(0)
dates = pd.date_range('1925-01-01', periods=95, freq='Y')
observed_data = np.random.rand(50)  # replace this with your actual data


# Fit an AR model with lag 1
model = AutoReg(observed_data, lags=1)
model_fit = model.fit()

# Generate 10 stochastic time series of 50 years each
num_series = 10
num_years = 50
generated_series = []

for _ in range(num_series):
    series = [observed_data[-1]]  # start with the last value of observed data
    for _ in range(num_years - 1):
        # Generate next value based on AR(1) model
        next_val = model_fit.params[0] + model_fit.params[1] * series[-1] + np.random.normal(scale=model_fit.resid.std())
        series.append(next_val)
    generated_series.append(series)

# Convert to DataFrame for easier handling
generated_series = pd.DataFrame(generated_series).T
stat = generated_series.apply(func='std',axis=0)
# Plot the generated series
for i in range(num_series):
    plt.plot(generated_series[i], label=f'Series {i+1}')
plt.legend()
plt.show()