import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from settings.settings import task_order

# Load the CSV file
file_path = "./results/combined_results.csv"
data = pd.read_csv(file_path)

# Filter data for test set results, considering specific datasets for SSP
test_data = data[data['data_split'].isin(['test', 'casp12', 'casp13', 'casp14', 'ts115', 'cb513'])]

# Function to filter and process data for given task and metric
def process_data(df, task, metric, ssp=None):
    if ssp:
        df_filtered = df[(df["task"] == task) & (df["metric"] == metric) & (df["ssp"] == ssp)].copy()
    else:
        df_filtered = df[(df["task"] == task) & (df["metric"] == metric)].copy()

    df_filtered.loc[:, "model"] = df_filtered["model"].replace({"ankh": "Ankh", "p2s": "TooT-PLM-P2S"})
    return df_filtered[["model", "value", "data_split"]]

# Define tasks and their corresponding metrics
tasks_metrics = {
    "fluorescence": "eval_spearmanr",
    "SSP": "eval_accuracy",
    "localization": "eval_mcc",
    "ionchannels": "eval_mcc",
    "mp": "eval_mcc",
    "transporters": "eval_mcc",
    "solubility": "eval_mcc",
}

# Process data for each task and metric, storing results in a dictionary
summary_results = {}
for task, metric in tasks_metrics.items():
    if task == "SSP":
        for ssp_type in ["ssp3", "ssp8"]:
            summary_results[f"{task}_{ssp_type}"] = process_data(test_data, task, metric, ssp_type)
    else:
        summary_results[task] = process_data(test_data, task, metric)

# Map the metric names to their desired format for display
metric_format_map = {
    "eval_spearmanr": "Spearman's ρ",
    "eval_mcc": "MCC",
    "eval_accuracy": "Accuracy",
}

# Consolidate results and prepare for plotting
consolidated_results = pd.DataFrame()
for task, results in summary_results.items():
    results["Task"] = task
    consolidated_results = pd.concat([consolidated_results, results], ignore_index=True)

consolidated_results["Metric"] = consolidated_results["Task"].apply(lambda x: metric_format_map[tasks_metrics.get(x.split("_")[0], "")])
consolidated_results["Task"] = consolidated_results["Task"].replace({"SSP_ssp3": "ssp3", "SSP_ssp8": "ssp8"})
consolidated_results["Task"] = pd.Categorical(consolidated_results["Task"], categories=task_order, ordered=True)
consolidated_results.sort_values(["Task"], inplace=True)

# Display the first few rows of the consolidated results to verify the structure
consolidated_results.head()