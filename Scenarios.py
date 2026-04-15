import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##### WIND SCENARIOS #####
# To produce realistic wind scenarios, historical data from Denmark found on Renewables Ninja is used as a base
# Data from renewables.ninja is given in by capacity factor, so is multiplied by the 500MW capacity
# 20 Days of data is taken where the 20 days are randomly selected from 2020-2024

# Load the historical wind data 
df_wind = pd.read_csv("group_26_assignment_2_REM/Wind_Data_DK2_Renewablesninja.csv")

# Parse time column
df_wind["time"] = pd.to_datetime(df_wind["time"])

# Keep only recent years (data frame starts in 1980, but we only want 2020-2024)
df_wind = df_wind[(df_wind["time"] >= "2020-01-01") & (df_wind["time"] < "2025-01-01")]

# Convert capacity factor to actual power output (MW)
df_wind["wind_MW"] = 500 * df_wind["DK02 (capacity factor)"]

# Create a date column and keep only complete days
df_wind["date"] = df_wind["time"].dt.date
hours_per_day = df_wind.groupby("date").size()
valid_dates = hours_per_day[hours_per_day == 24].index

df_wind = df_wind[df_wind["date"].isin(valid_dates)].copy()

# Randomly select 20 days
np.random.seed(42) # For reproducibility

selected_dates = np.random.choice(valid_dates, size=20, replace=False)
selected_dates = sorted(selected_dates)

# Filter selected days
wind_20_days = df_wind[df_wind["date"].isin(selected_dates)].copy()

wind_scenarios_list = []

for d in selected_dates:
    day_values = df_wind.loc[df_wind["date"] == d, "wind_MW"].to_numpy()
    wind_scenarios_list.append(day_values)

wind_scenarios = np.array(wind_scenarios_list)

print("Shape:", wind_scenarios.shape)   # should be (20, 24)
print("Selected dates:")
print(selected_dates)
print("First scenario:")
print(wind_scenarios[0])

# Plot all 20 scenarios
plt.figure(figsize=(12, 6)) 
for i in range(wind_scenarios.shape[0]):
    plt.plot(wind_scenarios[i], label=f"Day {i+1}")
plt.title("20 Wind Scenarios (500MW Capacity)")
plt.xlabel("Hour of Day")
plt.ylabel("Wind Power Output (MW)")
plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
plt.grid()
plt.show()

