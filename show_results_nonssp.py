import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from settings.settings import task_order

# Load the CSV file
file_path = "./results/combined_results.csv"
data = pd.read_csv(file_path)

# Filter data for cross-validation and test set results
cv_data = data[data["data_split"] == "cross-validation"]
test_data = data[data["data_split"] == "test"]

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
sns.set_palette("colorblind")

# Define tasks and their corresponding metrics
tasks_metrics = {
    "localization": ["eval_mcc", "eval_f1", "eval_recall", "eval_precision", "eval_accuracy"],
    "ionchannels": ["eval_mcc", "eval_f1", "eval_recall", "eval_precision", "eval_accuracy"],
    "mp": ["eval_mcc", "eval_f1", "eval_recall", "eval_precision", "eval_accuracy"],
    "transporters": ["eval_mcc", "eval_f1", "eval_recall", "eval_precision", "eval_accuracy"],
    "solubility": ["eval_mcc", "eval_f1", "eval_recall", "eval_precision", "eval_accuracy"],
}

classification_tasks = [
    "localization",
    "ionchannels",
    "mp",
    "transporters",
    "solubility",
]

# Function to filter and process data for given task and metric
def process_data(df, task, metric):
    df_filtered = df[(df["task"] == task) & (df["metric"] == metric)].copy()
    df_filtered.loc[:, "model"] = df_filtered["model"].replace(
        {"ankh": "Ankh", "p2s": "TooT-PLM-P2S"}
    )
    return df_filtered[["model", "value", "data_split"]]

# Function to compute mean ± std and p-value for cross-validation results
def compute_stats(results, metric):
    stats = {}
    for task in classification_tasks:
        task_results = process_data(results, task, metric)
        for model in task_results["model"].unique():
            model_results = task_results[task_results["model"] == model]
            mean = model_results["value"].mean()
            std = model_results["value"].std()
            stats[f"{task}_{model}"] = (mean, std)
    return stats

# Function to calculate p-value for statistical significance analysis
def compute_p_value(results, metric):
    p_values = {}
    for task in classification_tasks:
        task_results = process_data(results, task, metric)
        ankh_values = task_results[task_results["model"] == "Ankh"]["value"].values
        p2s_values = task_results[task_results["model"] == "TooT-PLM-P2S"]["value"].values
        if len(ankh_values) == len(p2s_values) and len(ankh_values) > 1:
            try:
                _, p_value = ttest_rel(ankh_values, p2s_values)
                p_values[task] = p_value
            except ValueError:
                p_values[task] = np.nan
        else:
            p_values[task] = np.nan
    return p_values

# Calculate averages for classification tasks
def calculate_average_metrics(results):
    avg_metrics = {}
    for model in results["model"].unique():
        model_data = results[results["model"] == model]
        avg_metrics[model] = model_data["value"].astype(float).mean()
    return avg_metrics

# Process data for each task and metric, storing results in a dictionary
summary_results_cv = {}
summary_results_test = {}
for task, metrics in tasks_metrics.items():
    for metric in metrics:
        summary_results_cv[f"{task}_{metric}"] = process_data(cv_data, task, metric)
        summary_results_test[f"{task}_{metric}"] = process_data(test_data, task, metric)

# Calculate average metrics for classification tasks
classification_metrics = ["eval_mcc", "eval_f1", "eval_recall", "eval_precision", "eval_accuracy"]
avg_metrics_cv = {metric: compute_stats(cv_data, metric) for metric in classification_metrics}
avg_metrics_test = {metric: compute_stats(test_data, metric) for metric in classification_metrics}
p_values = {metric: compute_p_value(cv_data, metric) for metric in classification_metrics}

# Calculate average metrics across all tasks for each metric
def calculate_average_across_tasks(metrics_dict, p_values_dict, tasks):
    average_metrics = {}
    average_p_values = {}
    for metric, task_values in metrics_dict.items():
        ankh_values = [task_values[f"{task}_Ankh"][0] for task in tasks]
        p2s_values = [task_values[f"{task}_TooT-PLM-P2S"][0] for task in tasks]
        average_metrics[f"Ankh_{metric}"] = (np.mean(ankh_values), np.std(ankh_values))
        average_metrics[f"TooT-PLM-P2S_{metric}"] = (np.mean(p2s_values), np.std(p2s_values))
        # Calculate p-value for the average metrics
        try:
            _, avg_p_value = ttest_rel(ankh_values, p2s_values)
            average_p_values[metric] = avg_p_value
        except ValueError:
            average_p_values[metric] = np.nan
    return average_metrics, average_p_values

average_cv_metrics, average_cv_p_values = calculate_average_across_tasks(avg_metrics_cv, p_values, classification_tasks)
average_test_metrics, average_test_p_values = calculate_average_across_tasks(avg_metrics_test, p_values, classification_tasks)

# Prepare data for LaTeX table
latex_table_data = []

# Add average metrics for each classification metric
for metric in classification_metrics:
    latex_table_data.append({
        "Metric": metric,
        "Model": "Ankh",
        "Cross-Validation": f"{average_cv_metrics[f'Ankh_{metric}'][0]:.4f} ± {average_cv_metrics[f'Ankh_{metric}'][1]:.4f}",
        "Test Set": f"{average_test_metrics[f'Ankh_{metric}'][0]:.4f}",
        "p-value": f"{average_cv_p_values[metric]:.4f}" if not np.isnan(average_cv_p_values[metric]) else "N/A"
    })
    latex_table_data.append({
        "Metric": metric,
        "Model": "TooT-PLM-P2S",
        "Cross-Validation": f"{average_cv_metrics[f'TooT-PLM-P2S_{metric}'][0]:.4f} ± {average_cv_metrics[f'TooT-PLM-P2S_{metric}'][1]:.4f}",
        "Test Set": f"{average_test_metrics[f'TooT-PLM-P2S_{metric}'][0]:.4f}",
        "p-value": f"{average_cv_p_values[metric]:.4f}" if not np.isnan(average_cv_p_values[metric]) else "N/A"
    })

latex_table = pd.DataFrame(latex_table_data)

# Generate LaTeX table
latex_table_code = latex_table.to_latex(
    index=False,
    caption="Average Performance Metrics of Ankh and TooT-PLM-P2S across Classification Tasks",
    label="tab:average_performance_metrics",
    escape=False,
)
print("LaTeX Table:")
print(latex_table_code)
