import numpy as np
import pandas as pd

def generate_scenarios():
    np.set_printoptions(suppress=True, formatter={'float_kind': '{:.2f}'.format})

    ##### WIND SCENARIOS #####
    df_wind = pd.read_csv("Wind_Data_DK2_Renewablesninja.csv")
    df_wind["time"] = pd.to_datetime(df_wind["time"])

    df_wind = df_wind[
        (df_wind["time"] >= "2020-01-01") &
        (df_wind["time"] < "2025-01-01")
    ]

    df_wind["wind_MW"] = 500 * df_wind["DK02 (capacity factor)"]
    df_wind["date"] = df_wind["time"].dt.date

    hours_per_day = df_wind.groupby("date").size()
    valid_dates = hours_per_day[hours_per_day == 24].index
    df_wind = df_wind[df_wind["date"].isin(valid_dates)].copy()

    np.random.seed(42)
    selected_dates_wind = np.random.choice(valid_dates, size=20, replace=False)

    wind_scenarios = np.array([
        df_wind.loc[df_wind["date"] == d, "wind_MW"].to_numpy()
        for d in selected_dates_wind
    ])

    ##### PRICE SCENARIOS #####
    df_price = pd.read_csv("GUI_ENERGY_PRICES_202312312300-202412312300.csv")

    df_price["time"] = df_price["MTU (CET/CEST)"].str.split(" - ").str[0]
    df_price["time"] = df_price["time"].str.replace(
        r"\s*\(CET\)|\s*\(CEST\)", "", regex=True
    )
    df_price["time"] = pd.to_datetime(df_price["time"], dayfirst=True)

    df_price = df_price.rename(columns={"Day-ahead Price (EUR/MWh)": "price"})
    df_price = df_price[["time", "price"]].dropna().copy()

    df_price["date"] = df_price["time"].dt.date

    counts = df_price.groupby("date").size()
    valid_dates_price = counts[counts == 24].index
    df_price = df_price[df_price["date"].isin(valid_dates_price)].copy()

    np.random.seed(42)
    selected_dates_price = np.random.choice(valid_dates_price, size=20, replace=False)

    price_scenarios = np.array([
        df_price.loc[df_price["date"] == d, "price"].to_numpy()
        for d in selected_dates_price
    ])

    ##### IMBALANCE SCENARIOS #####
    np.random.seed(999)
    imbalance_scenarios = np.random.binomial(
        n=1,
        p=0.5,
        size=(4, 24)
    )

    ##### COMBINED SCENARIOS #####
    combined_scenarios = []

    for i in range(wind_scenarios.shape[0]):
        for j in range(price_scenarios.shape[0]):
            for k in range(imbalance_scenarios.shape[0]):

                wind = wind_scenarios[i]
                price = price_scenarios[j]
                imbalance = imbalance_scenarios[k]

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

    return combined_scenarios


# Only runs when executing this file directly
if __name__ == "__main__":
    scenarios = generate_scenarios()
    print("Total scenarios:", len(scenarios))
    print("Example scenario:", scenarios[0])