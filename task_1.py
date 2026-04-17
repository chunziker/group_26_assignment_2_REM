import math
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

from Scenarios import generate_scenarios


#------------------------------------------
#Importing Scenarios
#------------------------------------------

scenarios = generate_scenarios()
print(len(scenarios))
print(scenarios[0])

#------------------------------------------

#------------------------------------------
# Step 0: Define some constants
#------------------------------------------
p_nom = 500 # Nominal installed capacity of the wind farm in [MW]

#------------------------------------------
#Task1.1: Offering Strategy Under a One-Price Market
#------------------------------------------
def task_1(scenarios):
    # Create a Gurobi model
    model = gp.Model("task_1_one_price")

    #Define some indices
    T = range(1, 25) # Hours in the day (1-24)
    W = range(len(scenarios)) # Scenarios indices

    # Decision variables for each hour (0-23)
    # production offfer in time t in the day ahead mareket from the wind faram
    offer = {}
    for t in T:
        offer[t] = model.addVar(lb=0, ub=p_nom,
                                vtype=GRB.CONTINUOUS,
                                name=f"p_DA_{t}")
    # ------------------------------------------
    # Objective: maximize total profit across scenarios
    # ------------------------------------------
    total_profit = gp.quicksum(
        gp.quicksum(
            scenarios[w]["price_DA"][t - 1] * offer[t]
            + scenarios[w]["price_balancing"][t - 1]
              * (scenarios[w]["wind"][t - 1] - offer[t])
            for t in T
        )
        for w in W
    )
    model.setObjective(total_profit, GRB.MAXIMIZE)

    model.optimize()

    optimal_offer = [offer[t].X for t in T]
    #Expected profit = average across scenarios 
    expected_profit = model.objVal / len(scenarios)

    # --------------------------------------------
    #profit per scenario
    # --------------------------------------------

    scenario_profits = []
    for w in W:
        profit = 0
        for t in T:
            p_da = scenarios[w]["price_DA"][t - 1]
            p_bal = scenarios[w]["price_balancing"][t - 1]
            wind = scenarios[w]["wind"][t - 1]
            bid  = optimal_offer[t-1]
            profit += p_da * bid + p_bal * (wind - bid)
        scenario_profits.append(profit)

    # ------------------------------------------
    # Convert to DataFrames
    # ------------------------------------------
    offer_df = pd.DataFrame({
        "hour": list(T),
        "offer_MW": [optimal_offer[t-1] for t in T]
    })

    profit_df = pd.DataFrame({
        "scenario": list(W),
        "profit_EUR": scenario_profits
    })

    return offer_df, expected_profit, profit_df
# ------------------------------------------
# Run model
# ------------------------------------------
offer_df_t1, expected_profit_t1, profit_df_t1 = task_1(scenarios)

print("\nOptimal hourly offers:")
print(offer_df_t1)

print(f"\nExpected profit: {expected_profit_t1:.2f} EUR")

print("\nProfit distribution:")
print(profit_df_t1.describe())


# ------------------------------------------
# Plot profit distribution
# ------------------------------------------
plt.figure()
plt.hist(profit_df_t1["profit_EUR"], bins=30)
plt.xlabel("Profit [EUR]")
plt.ylabel("Frequency")
plt.title("Profit distribution across scenarios")
plt.show()

#------------------------------------------
#Task1.2: Repeat Task 1.1 under the two-price scheme. Compare both offering strategies and profit
#distributions, and explain key differences.
#------------------------------------------
def task_2(scenarios):
    # Create a Gurobi model
    model = gp.Model("task_1_one_price")

    #Define some indices
    T = range(1, 25) # Hours in the day (1-24)
    W = range(len(scenarios)) # Scenarios indices

    # Decision variables for each hour (0-23)
    # production offfer in time t in the day ahead mareket from the wind faram
    offer = {}
    for t in T:
        offer[t] = model.addVar(lb=0, ub=p_nom,
                                vtype=GRB.CONTINUOUS,
                                name=f"p_DA_{t}")
    # ------------------------------------------
    # Objective: maximize total profit across scenarios
    # ------------------------------------------
    total_profit = gp.quicksum(
        gp.quicksum(
            scenarios[w]["price_DA"][t - 1] * offer[t]
            + (0.9 *scenarios[w]["price_balancing"][t - 1]
            * (scenarios[w]["wind"][t - 1] - offer[t]) 
            - 1.2 * scenarios[w]["price_balancing"][t - 1] * 
            (scenarios[w]["wind"][t - 1] - offer[t] )
            for t in T
        )
        for w in W
    )
    model.setObjective(total_profit, GRB.MAXIMIZE)

    model.optimize()

    optimal_offer = [offer[t].X for t in T]
    #Expected profit = average across scenarios 
    expected_profit = model.objVal / len(scenarios)

    # --------------------------------------------
    #profit per scenario
    # --------------------------------------------

    scenario_profits = []
    for w in W:
        profit = 0
        for t in T:
            p_da = scenarios[w]["price_DA"][t - 1]
            p_bal = scenarios[w]["price_balancing"][t - 1]
            wind = scenarios[w]["wind"][t - 1]
            bid  = optimal_offer[t-1]
            profit += p_da * bid + p_bal * (wind - bid)
        scenario_profits.append(profit)

    # ------------------------------------------
    # Convert to DataFrames
    # ------------------------------------------
    offer_df = pd.DataFrame({
        "hour": list(T),
        "offer_MW": [optimal_offer[t-1] for t in T]
    })

    profit_df = pd.DataFrame({
        "scenario": list(W),
        "profit_EUR": scenario_profits
    })

    return offer_df, expected_profit, profit_df
# ------------------------------------------
# Run model
# ------------------------------------------
offer_df_t1, expected_profit_t1, profit_df_t1 = task_1(scenarios)



