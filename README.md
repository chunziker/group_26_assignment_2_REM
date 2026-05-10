# Project README

## Overview
This project analyzes different electricity market strategies for a wind farm.  
The code uses wind production data, electricity prices, and imbalance scenarios to simulate and optimize market offers.

The project includes:
- Scenario generation
- One-price market optimization
- Two-price market optimization
- Cross-validation analysis
- Risk-averse offering strategies
---

## Folder Structure

```text
01_old/        Old scripts and previous versions
02_data/       Input datasets
03_results/    Output results and generated figures

Scenarios.py   Generates wind, price, and imbalance scenarios
Task_1.py      Main optimization and analysis tasks
Task_2.py      Reliability and CVaR bidding analysis
README.md      Project description
```

---

## Data
The project uses:
- Wind production data
- Day-ahead electricity prices
- Simulated imbalance scenarios

All input files are stored in the `02_data` folder.

---

## Main Files

### `Scenarios.py`
Creates combined scenarios for:
- Wind production
- Day-ahead prices
- Balancing prices
- System imbalance

### `Task_1.py`
Implements:
- One-price market model
- Two-price market model
- Cross-validation
- Risk-averse optimization

### `Task_2.py`
Implements:
- Load profile generation
- ALSO-X benchmark bidding
- CVaR bidding
- Reliability analysis

---

## How to Run

Run the scripts directly in Python:

```bash
python Task_1.py
```

or

```bash
python Task_2.py
```

---

## Results
Generated outputs are saved in:

```text
03_results/
```

This includes:
- CSV result files
- Figures
- Histograms
- Load profile plots

---

## Authors

- Bella Swan Cay
- Carlos Omar Hunziker
- Nikolay Yuliyanov Marinov 
- Izabella Kertész 
