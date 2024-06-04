import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

# Load the CSV file
file_path = "./results/combined_results.csv"
data = pd.read_csv(file_path)

# Filter data for cross-validation and test set results
cv_data = data[data["data_split"] == "cross-validation"]
test_data = data[
    data["data_split"].isin(["casp12", "casp13", "casp14", "ts115", "cb513"])
]

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
sns.set_palette("colorblind")

# Define tasks and their corresponding metrics
tasks_metrics = {
    "ssp3": ["eval_f1", "eval_recall", "eval_precision", "eval_accuracy"],
    "ssp8": ["eval_f1", "eval_recall", "eval_precision", "eval_accuracy"],
}

classification_tasks = ["ssp3", "ssp8"]

# Function to filter and process data for given task and metric
def process_data(df, task, metric):
    df_filtered = df[(df["ssp"] == task) & (df["metric"] == metric)].copy()
    df_filtered.loc[:, "model"] = df_filtered["model"].replace(
        {"ankh": "Ankh", "p2s": "TooT-PLM-P2S"}
    )
    return df_filtered[["model", "value", "data_split"]]

# Function to combine ssp3 and ssp8 data
def combine_tasks_data(results, metric):
    combined_data = pd.concat(
        [process_data(results, task, metric) for task in classification_tasks]
    )
    return combined_data

# Function to compute mean ± std and p-value for cross-validation results
def compute_stats(results, metric):
    stats = {}
    combined_results = combine_tasks_data(results, metric)
    for model in combined_results["model"].unique():
        model_results = combined_results[combined_results["model"] == model]
        mean = model_results["value"].mean()
        std = model_results["value"].std()
        stats[model] = (mean, std)
    return stats

# Function to calculate p-value for statistical significance analysis
def compute_p_value(results, metric):
    combined_results = combine_tasks_data(results, metric)
    ankh_values = combined_results[combined_results["model"] == "Ankh"]["value"].values
    p2s_values = combined_results[combined_results["model"] == "TooT-PLM-P2S"][
        "value"
    ].values
    if len(ankh_values) == len(p2s_values) and len(ankh_values) > 1:
        try:
            _, p_value = ttest_rel(ankh_values, p2s_values)
            return p_value
        except ValueError:
            return np.nan
    else:
        return np.nan

# Process data for each metric, storing results in a dictionary
summary_results_cv = {}
summary_results_test = {}
for task, metrics in tasks_metrics.items():
    for metric in metrics:
        summary_results_cv[metric] = combine_tasks_data(cv_data, metric)
        summary_results_test[metric] = combine_tasks_data(test_data, metric)

# Calculate average metrics for classification tasks
classification_metrics = ["eval_f1", "eval_recall", "eval_precision", "eval_accuracy"]
avg_metrics_cv = {metric: compute_stats(cv_data, metric) for metric in classification_metrics}
avg_metrics_test = {metric: compute_stats(test_data, metric) for metric in classification_metrics}
p_values = {metric: compute_p_value(cv_data, metric) for metric in classification_metrics}

# Prepare data for LaTeX table
latex_table_data = []

# Add average metrics for each classification metric
for metric in classification_metrics:
    latex_table_data.append(
        {
            "Metric": metric,
            "Model": "Ankh",
            "Cross-Validation": f"{avg_metrics_cv[metric]['Ankh'][0]:.4f} ± {avg_metrics_cv[metric]['Ankh'][1]:.4f}",
            "Test Set": f"{avg_metrics_test[metric]['Ankh'][0]:.4f}",
            "p-value": (
                f"{p_values[metric]:.4f}"
                if not np.isnan(p_values[metric])
                else "N/A"
            ),
        }
    )
    latex_table_data.append(
        {
            "Metric": metric,
            "Model": "TooT-PLM-P2S",
            "Cross-Validation": f"{avg_metrics_cv[metric]['TooT-PLM-P2S'][0]:.4f} ± {avg_metrics_cv[metric]['TooT-PLM-P2S'][1]:.4f}",
            "Test Set": f"{avg_metrics_test[metric]['TooT-PLM-P2S'][0]:.4f}",
            "p-value": (
                f"{p_values[metric]:.4f}"
                if not np.isnan(p_values[metric])
                else "N/A"
            ),
        }
    )

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
