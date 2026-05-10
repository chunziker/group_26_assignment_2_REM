import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
N_PROFILES = 300
N_MINUTES = 60
P_MIN = 220.0
P_MAX = 600.0
MAX_STEP = 35.0
EPS = 0.10

rng = np.random.default_rng(SEED)


def generate_profile():
    x = np.empty(N_MINUTES)
    x[0] = rng.uniform(P_MIN, P_MAX)
    for t in range(1, N_MINUTES):
        low = max(P_MIN, x[t - 1] - MAX_STEP)
        high = min(P_MAX, x[t - 1] + MAX_STEP)
        x[t] = rng.uniform(low, high)
    return x


def exact_p90_bid(A, eps=0.10):
    A_sorted = np.sort(A)
    k = int(np.floor(eps * len(A)))
    return float(A_sorted[k])


def cvar_left_side(c, A, eps=0.10):
    loss = c - A
    t_candidates = np.unique(loss)
    best = np.inf
    for t in t_candidates:
        value = t + np.mean(np.maximum(loss - t, 0.0)) / eps
        if value < best:
            best = value
    return best


def cvar_bid(A, eps=0.10, low=0.0, high=600.0, tol=1e-8):
    for _ in range(80):
        mid = 0.5 * (low + high)
        if cvar_left_side(mid, A, eps) <= 0:
            low = mid
        else:
            high = mid
        if high - low < tol:
            break
    return float(low)


def verify_bid(profiles, bid):
    min_load = profiles.min(axis=1)
    passes = min_load >= bid
    shortfall = np.maximum(bid - profiles, 0.0)
    max_shortfall = shortfall.max(axis=1)

    return {
        "bid_kW": float(bid),
        "profiles_passing": int(passes.sum()),
        "profiles_failing": int((~passes).sum()),
        "reliability": float(passes.mean()),
        "P90_met": bool(passes.mean() >= 0.90),
        "avg_max_shortfall_kW": float(max_shortfall.mean()),
        "avg_max_shortfall_if_fail_kW": float(max_shortfall[max_shortfall > 0].mean()) if np.any(max_shortfall > 0) else 0.0,
        "max_shortfall_kW": float(max_shortfall.max()),
        "shortfall_minutes": int((shortfall > 0).sum())
    }


profiles = np.vstack([generate_profile() for _ in range(N_PROFILES)])
in_sample = profiles[:100, :]
out_of_sample = profiles[100:, :]

assert profiles.shape == (300, 60)
assert np.all(profiles >= P_MIN)
assert np.all(profiles <= P_MAX)
assert np.all(np.abs(np.diff(profiles, axis=1)) <= MAX_STEP + 1e-9)

A_in = in_sample.min(axis=1)
A_out = out_of_sample.min(axis=1)
print("\nSupport summary")
print(f"In-sample support min / mean / max [kW]:  {A_in.min():.2f} / {A_in.mean():.2f} / {A_in.max():.2f}")
print(f"Out-of-sample support min / mean / max [kW]: {A_out.min():.2f} / {A_out.mean():.2f} / {A_out.max():.2f}")

print("\nSupport quantiles [kW]")
print(f"In-sample 10% / 50% / 90%:   {np.quantile(A_in, 0.10):.2f} / {np.quantile(A_in, 0.50):.2f} / {np.quantile(A_in, 0.90):.2f}")
print(f"Out-of-sample 10% / 50% / 90%:{np.quantile(A_out, 0.10):.2f} / {np.quantile(A_out, 0.50):.2f} / {np.quantile(A_out, 0.90):.2f}")


c_alsox = exact_p90_bid(A_in, EPS)
c_cvar = cvar_bid(A_in, EPS, low=0.0, high=P_MAX)
print("\nChosen bids for Task 2.1")
print(f"ALSO-X benchmark bid [kW]: {c_alsox:.2f}")
print(f"CVaR bid [kW]:             {c_cvar:.2f}")
print(f"Difference [kW]:           {c_alsox - c_cvar:.2f}")

print("\nP90 thresholds")
print(f"In-sample profiles required to pass:  {int((1-EPS)*len(in_sample))} out of {len(in_sample)}")
print(f"Out-of-sample profiles required to pass: {int((1-EPS)*len(out_of_sample))} out of {len(out_of_sample)}")


in_alsox = verify_bid(in_sample, c_alsox)
in_cvar = verify_bid(in_sample, c_cvar)

out_alsox = verify_bid(out_of_sample, c_alsox)
out_cvar = verify_bid(out_of_sample, c_cvar)

print("\nTask 2.1 check on in-sample data")
print(f"ALSO-X benchmark: {in_alsox['profiles_passing']}/{len(in_sample)} pass, reliability = {100*in_alsox['reliability']:.2f}%, P90 = {in_alsox['P90_met']}")
print(f"CVaR:             {in_cvar['profiles_passing']}/{len(in_sample)} pass, reliability = {100*in_cvar['reliability']:.2f}%, P90 = {in_cvar['P90_met']}")

print("\nTask 2.2 check on out-of-sample data")
print(f"ALSO-X benchmark: {out_alsox['profiles_passing']}/{len(out_of_sample)} pass, reliability = {100*out_alsox['reliability']:.2f}%, P90 = {out_alsox['P90_met']}")
print(f"CVaR:             {out_cvar['profiles_passing']}/{len(out_of_sample)} pass, reliability = {100*out_cvar['reliability']:.2f}%, P90 = {out_cvar['P90_met']}")

data_df = pd.DataFrame([{
    "profiles_total": N_PROFILES,
    "profiles_in_sample": in_sample.shape[0],
    "profiles_out_of_sample": out_of_sample.shape[0],
    "minutes_per_profile": N_MINUTES,
    "load_min_kW": P_MIN,
    "load_max_kW": P_MAX,
    "max_step_kW": MAX_STEP
}])

task21_df = pd.DataFrame([
    {
        "method": "ALSO-X benchmark",
        "bid_kW": c_alsox,
        "in_sample_reliability": in_alsox["reliability"],
        "P90_met_in_sample": in_alsox["P90_met"]
    },
    {
        "method": "CVaR",
        "bid_kW": c_cvar,
        "in_sample_reliability": in_cvar["reliability"],
        "P90_met_in_sample": in_cvar["P90_met"]
    }
])

task22_df = pd.DataFrame([
    {
        "method": "ALSO-X benchmark",
        "bid_kW": c_alsox,
        "out_of_sample_reliability": out_alsox["reliability"],
        "P90_met_out_of_sample": out_alsox["P90_met"],
        "profiles_passing": out_alsox["profiles_passing"],
        "profiles_failing": out_alsox["profiles_failing"],
        "avg_max_shortfall_kW": out_alsox["avg_max_shortfall_kW"],
        "avg_max_shortfall_if_fail_kW": out_alsox["avg_max_shortfall_if_fail_kW"],
        "max_shortfall_kW": out_alsox["max_shortfall_kW"],
        "shortfall_minutes": out_alsox["shortfall_minutes"]
    },
    {
        "method": "CVaR",
        "bid_kW": c_cvar,
        "out_of_sample_reliability": out_cvar["reliability"],
        "P90_met_out_of_sample": out_cvar["P90_met"],
        "profiles_passing": out_cvar["profiles_passing"],
        "profiles_failing": out_cvar["profiles_failing"],
        "avg_max_shortfall_kW": out_cvar["avg_max_shortfall_kW"],
        "avg_max_shortfall_if_fail_kW": out_cvar["avg_max_shortfall_if_fail_kW"],
        "max_shortfall_kW": out_cvar["max_shortfall_kW"],
        "shortfall_minutes": out_cvar["shortfall_minutes"]
    }
])

print("\nData")
print(data_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print("\nTask 2.1")
print(task21_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if x < 10 else f"{x:.2f}"))

print("\nTask 2.2")
print(task22_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if x < 10 else f"{x:.2f}"))

pd.DataFrame(in_sample).to_csv("task2_in_sample_profiles.csv", index=False)
pd.DataFrame(out_of_sample).to_csv("task2_out_of_sample_profiles.csv", index=False)
task21_df.to_csv("task2_1_results.csv", index=False)
task22_df.to_csv("task2_2_results.csv", index=False)

plt.figure(figsize=(9, 5))
for i in range(8):
    plt.plot(in_sample[i], alpha=0.8)
plt.xlabel("Minute")
plt.ylabel("Load [kW]")
plt.title("Sample of in-sample load profiles")
plt.grid(True)
plt.tight_layout()
plt.savefig("task2_profiles.png", dpi=300)
plt.show()

plt.figure(figsize=(9, 5))
plt.hist(A_in, bins=20, alpha=0.8, label="In-sample hourly support")
plt.axvline(c_alsox, linestyle="--", linewidth=2, label=f"ALSO-X benchmark = {c_alsox:.2f} kW")
plt.axvline(c_cvar, linestyle="--", linewidth=2, label=f"CVaR = {c_cvar:.2f} kW")
plt.xlabel("Hourly support min load [kW]")
plt.ylabel("Count")
plt.title("In-sample support and optimal bids")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("task2_bids_hist.png", dpi=300)
plt.show()

