import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from scipy.stats import linregress

# Load your data
dataset_path = "law.csv"
data = pd.read_csv(dataset_path)
halving_dates = [datetime(2012, 11, 28), datetime(2016, 7, 9), datetime(2020, 5, 11), datetime(2024, 4, 19), datetime(2028, 4, 19)]

# Convert 'Start' to datetime and 'Close' to numeric
data['Date'] = pd.to_datetime(data['Start'])
data['Price'] = pd.to_numeric(data['Close'], errors='coerce')
data = data.dropna(subset=['Date', 'Price'])
data = data.sort_values(by='Date')




# --- 1. Calculate the Linear Regression ---
time_numeric = mdates.date2num(data['Date'])
log_price = np.log(data['Price'])
slope, intercept, r_value, p_value, std_err = linregress(time_numeric, log_price)
linear_regression_fit = np.exp(intercept + slope * time_numeric)

# --- 2. Define Support and Resistance Bands (Illustrative) ---
std_dev = np.std(log_price - (intercept + slope * time_numeric))
upper_band = np.exp(intercept + (2 * std_dev) + slope * time_numeric)
lower_band = np.exp(intercept - (2 * std_dev) + slope * time_numeric)




# --- 3. Plotting ---
plt.figure(figsize=(14, 8))
plt.style.use('dark_background')


# Plot Bitcoin Price, Linear Reg, Upper and Lower Band, and Halving Markers 
plt.plot(data['Date'], data['Price'], color='orange', label='Price History')
plt.plot(data['Date'], linear_regression_fit, color='red', linestyle='-', label='Linear Regression')
plt.plot(data['Date'], upper_band, color='purple', linestyle='--', alpha=0.7, label='Resistance')
plt.plot(data['Date'], lower_band, color='green', linestyle='--', alpha=0.7, label='Support')
for date in halving_dates:
    plt.axvline(date, color='red', linestyle='--', alpha=0.5, label='Halving' if date == halving_dates[0] else "")


plt.yscale('log')
plt.xlim(data['Date'].min() - pd.Timedelta(days=365), datetime(2025, 1, 1)) 
years = mdates.YearLocator(base=1)
years_fmt = mdates.DateFormatter('%Y')
plt.gca().xaxis.set_major_locator(years)
plt.gca().xaxis.set_major_formatter(years_fmt)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Year')
plt.ylabel('BTC Price ($)')
plt.title('Bitcoin Price with Linear Regression and Potential Bands')
plt.tick_params(axis='x', colors='black')
plt.tick_params(axis='y', colors='black')
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.1)
plt.tight_layout()
plt.show()