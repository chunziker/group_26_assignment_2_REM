import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Display prices as normal numbers, not scientific notation
np.set_printoptions(suppress=True, formatter={'float_kind':'{:.2f}'.format})

##### WIND SCENARIOS #####
# To produce realistic wind scenarios, historical data from Denmark found on Renewables Ninja is used as a base
# Data from renewables.ninja is given in by capacity factor, so is multiplied by the 500MW capacity
# 20 Days of data is taken where the 20 days are randomly selected from 2020-2024

# Load the historical wind data 
df_wind = pd.read_csv("Wind_Data_DK2_Renewablesninja.csv")

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

### PRICE SCENARIOS ###
# entsoe data is used to create price scenarios 
# Randomly selected 20 days from the year 2024 (last full year of data with hour resolution)

# Load the historical price data
df_price = pd.read_csv("GUI_ENERGY_PRICES_202312312300-202412312300.csv")

# Parse time column
# Extract start time of interval
df_price["time"] = df_price["MTU (CET/CEST)"].str.split(" - ").str[0]
# Remove literal timezone tags like " (CET)" or " (CEST)"
df_price["time"] = df_price["time"].str.replace(r"\s*\(CET\)|\s*\(CEST\)", "", regex=True)
# Convert to datetime
df_price["time"] = pd.to_datetime(df_price["time"], dayfirst=True)

# Keep only Day-ahead prices
df_price = df_price.rename(columns={"Day-ahead Price (EUR/MWh)": "price"})
df_price = df_price[["time", "price"]].dropna().copy()

# Extract date 
df_price["date"] = df_price["time"].dt.date
# Keep only complete days
counts = df_price.groupby("date").size()
valid_dates = counts[counts == 24].index

df_price = df_price[df_price["date"].isin(valid_dates)].copy()

# Randomly select 20 days
np.random.seed(42) # For reproducibility
selected_dates = np.random.choice(valid_dates, size=20, replace=False)

price_scenarios_list = []

for d in selected_dates:
    day_values = df_price.loc[df_price["date"] == d, "price"].to_numpy()
    price_scenarios_list.append(day_values)

price_scenarios = np.array(price_scenarios_list)

# Check
print("Shape:", price_scenarios.shape)  # should be (20, 24)
print("Selected dates:")
print(selected_dates)

#### SYSTEM IMBALANCE SCENARIOS ####
# 1 = Deficit
# 0 = Surplus
np.random.seed(999)   # fixed seed for reproducibility

n_imbalance_scenarios = 4
n_hours = 24
p_deficit = 0.5       # probability of deficit

imbalance_scenarios = np.random.binomial(
    n=1,
    p=p_deficit,
    size=(n_imbalance_scenarios, n_hours)
)

print("Shape:", imbalance_scenarios.shape)   # should be (4, 24)
print(imbalance_scenarios)

### COMBINED SCENARIOS ###

combined_scenarios = []

for i in range(wind_scenarios.shape[0]):
    for j in range(price_scenarios.shape[0]):
        for k in range(imbalance_scenarios.shape[0]):

            wind = wind_scenarios[i]              # (24,)
            price = price_scenarios[j]            # (24,)
            imbalance = imbalance_scenarios[k]    # (24,)

            # Balancing price
            balancing_price = np.where(
                imbalance == 1,
                1.25 * price,
                0.85 * price
            )

            scenario = {
                "wind": wind,
                "price_DA": price,
                "imbalance": imbalance,
                "price_balancing": balancing_price
            }

            combined_scenarios.append(scenario)

print("Total scenarios:", len(combined_scenarios))   # should be 1600
print ("Example scenario:")
print(combined_scenarios[0])