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

#------------------------------------------
#Task1.2: Repeat Task 1.1 under the two-price scheme. Compare both offering strategies and profit
#distributions, and explain key differences.
#------------------------------------------
def task_2(scenarios):
    # Create a Gurobi model
    model = gp.Model("task_2_two_price")

    # Define some indices
    T = range(1, 25)  # Hours in the day (1-24)
    W = range(len(scenarios))  # Scenario indices

    # Decision variables for each hour
    offer = {}
    for t in T:
        offer[t] = model.addVar(lb=0, ub=p_nom,
                                vtype=GRB.CONTINUOUS,
                                name=f"p_DA_{t}")

    # Auxiliary variables for excess and deficit (linearization)
    excess = {}
    deficit = {}
    for w in W:
        for t in T:
            excess[w, t] = model.addVar(lb=0,ub=p_nom, vtype=GRB.CONTINUOUS,
                                        name=f"excess_{w}_{t}")
            deficit[w, t] = model.addVar(lb=0, ub=p_nom, vtype=GRB.CONTINUOUS,
                                         name=f"deficit_{w}_{t}")

    # Imbalance split: wind - offer = excess - deficit
    for w in W:
        for t in T:
            model.addConstr(
                scenarios[w]["wind"][t - 1] - offer[t] == excess[w, t] - deficit[w, t],
                name=f"imbalance_split_{w}_{t}"
            )

    # Objective: maximize total profit across scenarios
    total_profit = gp.quicksum(
        gp.quicksum(
            scenarios[w]["price_DA"][t - 1] * offer[t]
            + 0.9 * scenarios[w]["price_DA"][t - 1] * excess[w, t]
            - 1.2 * scenarios[w]["price_DA"][t - 1] * deficit[w, t]
            for t in T
        )
        for w in W
    )
    model.setObjective(total_profit, GRB.MAXIMIZE)

    model.optimize()

    optimal_offer = [offer[t].X for t in T]
    expected_profit = model.objVal / len(scenarios)

    # Profit per scenario
    scenario_profits = []
    for w in W:
        profit = 0
        for t in T:
            p_da = scenarios[w]["price_DA"][t - 1]
            wind = scenarios[w]["wind"][t - 1]
            bid = optimal_offer[t - 1]

            excess_val = max(wind - bid, 0)
            deficit_val = max(bid - wind, 0)

            profit += p_da * bid + 0.9 * p_da * excess_val - 1.2 * p_da * deficit_val
        scenario_profits.append(profit)

    # Convert to DataFrames
    offer_df = pd.DataFrame({
        "hour": list(T),
        "offer_MW": [optimal_offer[t - 1] for t in T]
    })

    profit_df = pd.DataFrame({
        "scenario": list(W),
        "profit_EUR": scenario_profits
    })

    return offer_df, expected_profit, profit_df

def task_3(scenarios):
    # ------------------------------------------
    # Basic settings
    # ------------------------------------------
    T = range(1, 25)              # 24 hourly time periods
    n_folds = 8                   # 8-fold cross-validation
    fold_size = 200               # each fold has 200 scenarios
    n_total = len(scenarios)      # total number of scenarios (should be 1600)

    # Sanity check: assignment requirement
    if n_total != 1600:
        raise ValueError(f"task_3 expects exactly 1600 scenarios, but got {n_total}.")

    # ------------------------------------------
    # Shuffle scenarios (important for CV)
    # ------------------------------------------
    # We randomly permute indices so that folds are unbiased
    # Fixing the seed ensures reproducibility
    rng = np.random.default_rng(42)
    shuffled_indices = rng.permutation(n_total)

    # ------------------------------------------
    # Split scenarios into 8 folds (each size 200)
    # ------------------------------------------
    # Each fold will act once as "in-sample"
    # and the remaining 7 folds (1400 scenarios) as "out-of-sample"
    folds = []
    for k in range(n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size
        fold_idx = shuffled_indices[start:end]
        folds.append([scenarios[i] for i in fold_idx])

    # ------------------------------------------
    # Helper: evaluate a fixed offer on scenarios
    # ------------------------------------------
    # IMPORTANT:
    # - No optimization here!
    # - We just compute profits given a fixed bidding strategy
    def evaluate_offer(offer_df, eval_scenarios):

        # Convert DataFrame (output of task_2) into a simple list
        # so we can index it by hour
        optimal_offer = offer_df["offer_MW"].tolist()

        scenario_profits = []

        # Loop over all evaluation scenarios
        for w in range(len(eval_scenarios)):
            profit = 0

            # Loop over all hours
            for t in T:
                p_da = eval_scenarios[w]["price_DA"][t - 1]   # day-ahead price
                wind = eval_scenarios[w]["wind"][t - 1]       # realized wind
                bid = optimal_offer[t - 1]                    # fixed offer

                # Split imbalance into excess and deficit
                # This corresponds to the two-price scheme linearization
                excess_val = max(wind - bid, 0)   # overproduction
                deficit_val = max(bid - wind, 0)  # underproduction

                # Profit calculation (same as in task_2)
                profit += (
                    p_da * bid
                    + 0.9 * p_da * excess_val
                    - 1.2 * p_da * deficit_val
                )

            scenario_profits.append(profit)

        # Expected profit = average over all scenarios
        expected_profit = sum(scenario_profits) / len(eval_scenarios)

        return expected_profit, scenario_profits

    # ------------------------------------------
    # Store results
    # ------------------------------------------
    run_results = []   # one row per fold
    offer_rows = []    # store all optimal offers (for analysis)

    # ------------------------------------------
    # Cross-validation loop
    # ------------------------------------------
    for k in range(n_folds):

        # ------------------------------------------
        # Define in-sample and out-of-sample sets
        # ------------------------------------------
        in_sample = folds[k]   # current fold = training set (200 scenarios)

        # all other folds = test set (1400 scenarios)
        out_of_sample = [
            s for j in range(n_folds) if j != k for s in folds[j]
        ]

        # ------------------------------------------
        # STEP 1: Optimize on in-sample scenarios
        # ------------------------------------------
        # We reuse your task_2 model here
        # -> solves stochastic optimization problem
        offer_df, expected_profit_in, _ = task_2(in_sample)

        # ------------------------------------------
        # STEP 2: Evaluate same offer on out-of-sample
        # ------------------------------------------
        # IMPORTANT:
        # - No re-optimization!
        # - We test how well the decision generalizes
        expected_profit_out, _ = evaluate_offer(offer_df, out_of_sample)

        # ------------------------------------------
        # Store results for this fold
        # ------------------------------------------
        run_results.append({
            "run": k + 1,
            "n_in_sample": len(in_sample),
            "n_out_of_sample": len(out_of_sample),
            "expected_profit_in_sample_EUR": expected_profit_in,
            "expected_profit_out_of_sample_EUR": expected_profit_out
        })

        # Also store the hourly offer (useful for later analysis/plots)
        for _, row in offer_df.iterrows():
            offer_rows.append({
                "run": k + 1,
                "hour": int(row["hour"]),
                "offer_MW": row["offer_MW"]
            })

    # ------------------------------------------
    # Convert results to DataFrames
    # ------------------------------------------
    cv_results_df = pd.DataFrame(run_results)
    offers_df = pd.DataFrame(offer_rows)

    # ------------------------------------------
    # Compute summary statistics
    # ------------------------------------------
    # This is what you will use for interpretation
    summary_df = pd.DataFrame({
        "metric": [
            "average_in_sample_expected_profit_EUR",
            "average_out_of_sample_expected_profit_EUR",
            "difference_in_minus_out_EUR"
        ],
        "value": [
            cv_results_df["expected_profit_in_sample_EUR"].mean(),
            cv_results_df["expected_profit_out_of_sample_EUR"].mean(),
            cv_results_df["expected_profit_in_sample_EUR"].mean()
            - cv_results_df["expected_profit_out_of_sample_EUR"].mean()
        ]
    })

    # ------------------------------------------
    # Return all relevant outputs
    # ------------------------------------------
    return cv_results_df, summary_df, offers_df

def run_task_1():
    offer_df_t1, expected_profit_t1, profit_df_t1 = task_1(scenarios)

    print("\nOptimal hourly offers (Task 1):")
    print(offer_df_t1)

    print(f"\nExpected profit: {expected_profit_t1:.2f} EUR")

    print("\nProfit distribution:")
    print(profit_df_t1.describe())

    plt.figure()
    plt.hist(profit_df_t1["profit_EUR"], bins=30)
    plt.xlabel("Profit [EUR]")
    plt.ylabel("Frequency")
    plt.title("Task 1: Profit distribution across scenarios")
    plt.show()

    return offer_df_t1, expected_profit_t1, profit_df_t1


def run_task_2():
    offer_df_t2, expected_profit_t2, profit_df_t2 = task_2(scenarios)

    print("\nOptimal hourly offers (Task 2):")
    print(offer_df_t2)

    print(f"\nExpected profit: {expected_profit_t2:.2f} EUR")

    print("\nProfit distribution:")
    print(profit_df_t2.describe())

    plt.figure()
    plt.hist(profit_df_t2["profit_EUR"], bins=30)
    plt.xlabel("Profit [EUR]")
    plt.ylabel("Frequency")
    plt.title("Task 2: Profit distribution across scenarios")
    plt.show()

    return offer_df_t2, expected_profit_t2, profit_df_t2


def run_task_3():
    cv_results_df, summary_df, offers_df = task_3(scenarios)

    print("\nCross-validation results (per run):")
    print(cv_results_df)

    print("\nSummary (averages):")
    print(summary_df)

    plt.figure()
    plt.plot(
        cv_results_df["run"],
        cv_results_df["expected_profit_in_sample_EUR"],
        label="In-sample"
    )
    plt.plot(
        cv_results_df["run"],
        cv_results_df["expected_profit_out_of_sample_EUR"],
        label="Out-of-sample"
    )
    plt.xlabel("Fold")
    plt.ylabel("Expected Profit [EUR]")
    plt.title("Task 3: Cross-validation results")
    plt.legend()
    plt.show()

    return cv_results_df, summary_df, offers_df

if __name__ == "__main__":
    print("\nAvailable commands:")
    #print("run_task_1()")
    #print("run_task_2()")
    run_task_3()