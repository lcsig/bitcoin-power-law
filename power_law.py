import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

# Load your data
dataset_path = "law.csv"
data = pd.read_csv(dataset_path)
halving_dates = [datetime(2012, 11, 28), datetime(2016, 7, 9), datetime(2020, 5, 11), datetime(2024, 4, 19), datetime(2028, 4, 19)]

# Convert 'Start' to datetime and 'Close' to numeric
data['Date'] = pd.to_datetime(data['Start'])
data['Price'] = pd.to_numeric(data['Close'], errors='coerce')
data = data.dropna(subset=['Date', 'Price'])
data = data.sort_values(by='Date')



# --- 1. Calculate the Power Law Fit ---
def power_law(t, a, b):
    return a * (t ** b)

time_numeric = (data['Date'] - data['Date'].min()).dt.days.values
try:
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(power_law, time_numeric, data['Price'], p0=[1, 1])
    power_law_fit = power_law(time_numeric, *popt)
except ImportError:
    print("Error: scipy not installed.")
    power_law_fit = np.full_like(data['Price'], np.nan)
except RuntimeError:
    print("Error: Optimal parameters not found.")
    power_law_fit = np.full_like(data['Price'], np.nan)



# --- 2. Plotting ---
plt.figure(figsize=(14, 8))
plt.style.use('dark_background')


# Plot Bitcoin Price, Plot Power Law Fit, Plot Halving Markers
plt.plot(data['Date'], data['Price'], color='lime', label='Bitcoin Price')
plt.plot(data['Date'], power_law_fit, color='magenta', linestyle='-', label='Power Law Fit')
for date in halving_dates:
    plt.axvline(date, color='red', linestyle='--', alpha=0.7, label='Halving' if date == halving_dates[0] else "")


plt.yscale('log')                                                   # Set y-axis to logarithmic scale
plt.xlim(datetime(2010, 1, 1), datetime(2023, 12, 31))              # Set x-axis limits
year_dates = [datetime(year, 1, 1) for year in range(2010, 2024)]   # Create the desired year ticks
plt.xticks(year_dates, color='gray', rotation=45, ha='right')      # Set the x-axis ticks and labels with black color
plt.ylim(10**-2)                                                    # Set yLimit 
ax = plt.gca()                                                      # Get the current axes
ax.tick_params(axis='y', color='gray')                            # Set the y-axis tick labels color to black
ax.set_xlabel('Year', color='gray')                               
ax.set_ylabel('BTC Price ($)', color='gray')
ax.set_title('Bitcoin Power Law Chart', color='gray')
plt.yticks(color='gray')
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.1)
plt.tight_layout()
plt.show()