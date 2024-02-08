import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import numpy as np
import matplotlib.gridspec as gridspec

# Load the uploaded results file
file_path = "./results/combined_results.csv"
data = pd.read_csv(file_path)

# Filter data for the "transporters Prediction" task
transporters_data = data[data["task"] == "transporters"]

# Further filter for cross-validation data split
transporters_cv_data = transporters_data[
    transporters_data["data_split"] == "cross-validation"
]
# Group by model and metric to summarize the performance across folds
transporters_summary = (
    transporters_cv_data.groupby(["model", "metric"])["value"]
    .agg(["mean", "std"])
    .reset_index()
)

# Focus on key metrics for comparison
key_metrics = ["eval_accuracy", "eval_precision", "eval_recall", "eval_f1", "eval_mcc"]

metric_order = {metric: i for i, metric in enumerate(key_metrics)}


transporters_summary_filtered = transporters_summary[
    transporters_summary["metric"].isin(key_metrics)
]


# Function to perform statistical analysis and compute p-value
def compute_p_value(df, task, metric, ssp=None):
    if ssp:
        df_filtered = df[
            (df["task"] == task) & (df["metric"] == metric) & (df["ssp"] == ssp)
        ]
    else:
        df_filtered = df[(df["task"] == task) & (df["metric"] == metric)]

    df_filtered = df_filtered.copy()

    # Separate values for each model
    ankh_values = df_filtered[df_filtered["model"] == "ankh"]["value"].values
    p2s_values = df_filtered[df_filtered["model"] == "p2s"]["value"].values

    # Ensure there is a pair of results for each fold before performing the test
    if len(ankh_values) == len(p2s_values) and len(ankh_values) > 1:
        try:
            # Use paired t-test
            _, p_value = ttest_rel(ankh_values, p2s_values)
        except ValueError:
            # In case of an error (e.g., all values are identical), set p-value to NaN
            p_value = np.nan
    else:
        # If there's not a matching number of results for each model, we can't compute the p-value
        p_value = np.nan

    return p_value


# Function to process data and compute mean, std, and p-value for each metric
def process_and_compare_data(df, task, metrics):
    results = []
    p_values = {}

    for metric in metrics:
        # Filter data by task and metric
        df_filtered = df[(df["task"] == task) & (df["metric"] == metric)].copy()

        # Replace model codes with full names
        df_filtered["model"] = df_filtered["model"].replace(
            {"ankh": "Ankh", "p2s": "TooT-PLM-P2S"}
        )

        # Compute mean and std for each model
        summary = (
            df_filtered.groupby("model")["value"].agg(["mean", "std"]).reset_index()
        )

        # Add metric name (without 'eval_') for plotting
        summary["metric"] = metric.replace("eval_", "")

        results.append(summary)

        # Compute p-value
        ankh_values = df_filtered[df_filtered["model"] == "Ankh"]["value"].values
        p2s_values = df_filtered[df_filtered["model"] == "TooT-PLM-P2S"]["value"].values

        if len(ankh_values) == len(p2s_values) and len(ankh_values) > 1:
            _, p_value = ttest_rel(ankh_values, p2s_values)
            # Format the p-value with one decimal in scientific notation
            p_values[metric.replace("eval_", "")] = np.format_float_scientific(
                p_value, precision=1
            )
        else:
            p_value = np.nan
            p_values[metric.replace("eval_", "")] = np.format_float_scientific(
                p_value, precision=1
            )  # Handle NaN case

    return pd.concat(results), p_values


# Process data and compute p-values for the "transporters Prediction" task
transporters_results, transporters_p_values = process_and_compare_data(
    transporters_cv_data, "transporters", key_metrics
)

# Visualizing the results with error bars and p-values
fig = plt.figure(figsize=(12, 12))  # Adjust the figure size as needed

# Create a GridSpec layout
gs = gridspec.GridSpec(3, 2, figure=fig)  # 3 rows, 2 columns

# Iterate through the metrics and create subplots
for idx, metric in enumerate(key_metrics):
    if idx < 4:  # For the first four metrics, arrange in the first 2 rows
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
    else:  # For the last metric, center it in the last row
        ax = fig.add_subplot(gs[2, :])

    metric_name = metric.replace("eval_", "")
    subset = transporters_results[transporters_results["metric"] == metric_name]

    subset["order"] = subset["metric"].map(metric_order)
    subset = subset.sort_values(by="order")

    barplot = sns.barplot(
        x="metric",
        y="mean",
        hue="model",
        data=subset,
        palette="coolwarm",
        capsize=0.1,
        errwidth=1.5,
    )

    for bar, std in zip(barplot.patches, subset["std"]):
        ax.errorbar(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            yerr=std,
            ecolor="black",
            capsize=5,
            elinewidth=1.5,
        )

    ax.set_xlabel("")  # Remove the x-axis label
    ax.set_xticklabels([])  # Remove x-axis tick labels
    plt.title(metric_name.capitalize())
    plt.ylabel("Value")
    plt.ylim(0, 1)  # Adjust based on your data

    # Adjusting p-value annotation to be above the highest bar in the subplot
    highest_bar = max(subset["mean"]) + max(subset["std"])
    ax.text(
        0,
        highest_bar + 0.05,  # Add an offset
        f"p-value: {transporters_p_values.get(metric_name, np.nan)}",
        ha="center",
        va="bottom",
        fontsize=12,
    )

    # Move the legend to the top right of the plot
    ax.legend(title="Model", loc="upper right")

    if idx == 4:  # Break the loop after placing the last plot
        break

# plt.suptitle("transporters Prediction Performance Comparison", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

plt.savefig("./plots/plot3.2.5_trans.png", bbox_inches="tight", dpi=300)
plt.close()


# Preparing data for LaTeX table
transporters_results["Mean ± Std"] = transporters_results.apply(
    lambda row: f"{row['mean']:.3f} ± {row['std']:.3f}", axis=1
)
transporters_results["P-Value"] = transporters_results["metric"].map(transporters_p_values)

# Generating LaTeX table
transporters_latex_table = transporters_results.pivot(
    index="metric", columns="model", values=["Mean ± Std", "P-Value"]
).reset_index()
# Flatten the multi-level column index
transporters_latex_table.columns = [
    " ".join(col).strip() for col in transporters_latex_table.columns.values
]

# Print current column names to verify the structure
print("Current column names:", transporters_latex_table.columns.tolist())

# Based on the printed column names, create an accurate list of new column names
# Ensure this list matches the number of columns in your DataFrame
# Define new column names based on the current structure
new_column_names = [
    "Metric",
    "Ankh Mean ± Std",
    "TooT-PLM-P2S Mean ± Std",
    "Ankh P-Value",
    "TooT-PLM-P2S P-Value",
]

# Rename the columns using the new names
transporters_latex_table.columns = new_column_names

# Initialize a list to store the p-values for each metric
p_values = []

# Add p-values to the transporters_results DataFrame
for metric in key_metrics:
    metric_name = metric.replace("eval_", "")
    p_value = compute_p_value(transporters_cv_data, "transporters", metric)
    # Format the p-value with one decimal in scientific notation
    p_value_formatted = np.format_float_scientific(p_value, precision=1)
    transporters_results.loc[transporters_results["metric"] == metric_name, "P-Value"] = (
        p_value_formatted
    )

transporters_results["metric"] = transporters_results["metric"].apply(
    lambda x: "eval_" + x if not x.startswith("eval_") else x
)


# Ensure the DataFrame is sorted by the metric order before pivoting
transporters_results["metric_order"] = transporters_results["metric"].map(
    lambda x: key_metrics.index(x)
)
transporters_results.sort_values(by="metric_order", inplace=True)

# Pivot the table to get the desired layout
transporters_latex_table = transporters_results.pivot_table(
    index="metric",
    columns="model",
    values=["Mean ± Std", "P-Value"],
    aggfunc=lambda x: x,
).reset_index()

# Flatten the multi-level column index and sort by metric order
transporters_latex_table.columns = [
    " ".join(col).strip() for col in transporters_latex_table.columns.values
]
transporters_latex_table = transporters_latex_table.set_index("metric")
transporters_latex_table = transporters_latex_table.reindex(
    key_metrics
)  # Reorder based on key_metrics
transporters_latex_table.reset_index(inplace=True)

# Initialize the transposed DataFrame
transposed_df = pd.DataFrame(
    index=["Ankh", "TooT-PLM-P2S", "P-Value"],
    columns=[m.replace("eval_", "") for m in key_metrics],
)

# Fill in the mean ± std values
for metric in key_metrics:
    metric_name = metric.replace("eval_", "")
    for model in ["Ankh", "TooT-PLM-P2S"]:
        mean_std = transporters_results[
            (transporters_results["metric"] == metric)
            & (transporters_results["model"] == model)
        ]["Mean ± Std"].values[0]
        transposed_df.loc[model, metric_name] = mean_std

# Add p-values
for metric in key_metrics:
    metric_name = metric.replace("eval_", "")
    p_value = transporters_p_values[metric_name]
    transposed_df.loc["P-Value", metric_name] = p_value

# Convert the DataFrame to LaTeX
latex_code_transposed = transposed_df.to_latex(
    index=True,
    caption="transporters Prediction Performance Comparison Transposed with P-Values",
    label="tab:transporters_comparison_transposed_p_values",
    column_format="l" + "c" * len(key_metrics),  # One column for each metric
    float_format="%.3f",
    escape=False,
)

print(latex_code_transposed)
