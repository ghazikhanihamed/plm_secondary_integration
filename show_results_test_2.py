import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from settings.settings import task_order

# Load the CSV file
file_path = "./results/combined_results.csv"
data = pd.read_csv(file_path)

# Filter data for test set results, considering specific datasets for SSP
test_data = data[
    data["data_split"].isin(["test", "casp12", "casp13", "casp14", "ts115", "cb513"])
]

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
sns.set_palette("colorblind")


# Function to filter and process data for given task and metric
def process_data(df, task, metric, ssp=None):
    if ssp:
        df_filtered = df[
            (df["task"] == task) & (df["metric"] == metric) & (df["ssp"] == ssp)
        ].copy()
    else:
        df_filtered = df[(df["task"] == task) & (df["metric"] == metric)].copy()

    df_filtered.loc[:, "model"] = df_filtered["model"].replace(
        {"ankh": "Ankh", "p2s": "TooT-PLM-P2S"}
    )
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
            summary_results[f"{task}_{ssp_type}"] = process_data(
                test_data, task, metric, ssp_type
            )
    else:
        summary_results[task] = process_data(test_data, task, metric)

# Sorting order for SSP types
# sort_order = ["cb513", "ts115", "casp12", "casp13", "casp14"]

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
    results["Metric"] = results["Task"].apply(
        lambda x: metric_format_map[tasks_metrics.get(x.split("_")[0], "")]
    )
    consolidated_results = pd.concat([consolidated_results, results], ignore_index=True)

# Ensure all tasks are strings
consolidated_results["Task"] = consolidated_results["Task"].astype(str)


# Format values: round and convert to percentage if necessary
def format_values(row):
    return f"{row['value']:.4f}"


consolidated_results["value"] = consolidated_results.apply(format_values, axis=1)

consolidated_results["Task"] = consolidated_results["Task"].replace(
    {"SSP_ssp3": "ssp3", "SSP_ssp8": "ssp8"}
)

# Average and standard deviation for SSP tasks
ssp_avg_std = (
    consolidated_results[consolidated_results["Task"].str.contains("ssp")]
    .groupby(["Task", "model", "Metric"])
    .agg(
        mean_value=("value", lambda x: np.mean([float(i) for i in x])),
        std_value=("value", lambda x: np.std([float(i) for i in x])),
    )
    .reset_index()
)

# Combine mean and standard deviation into one column
ssp_avg_std["value"] = ssp_avg_std.apply(lambda row: f"{row['mean_value']:.4f}", axis=1)

# Update consolidated results for SSP tasks to use average and standard deviation
consolidated_results = consolidated_results[
    ~consolidated_results["Task"].str.contains("ssp")
]
consolidated_results = pd.concat(
    [consolidated_results, ssp_avg_std[["Task", "model", "Metric", "value"]]],
    ignore_index=True,
)

# Generate LaTeX table for all tasks
consolidated_results["Task"] = pd.Categorical(
    consolidated_results["Task"], categories=task_order, ordered=True
)
consolidated_results = consolidated_results.sort_values(["Task"])

latex_table = consolidated_results[["Task", "model", "Metric", "value"]].to_latex(
    index=False,
    caption="Comparative Performance Overview of Ankh and TooT-PLM-P2S across Various Tasks",
    label="tab:performance_comparison",
    escape=False,
)

print("LaTeX Table:")
print(latex_table)

# Visualization

# Combine SSP average with non-SSP data for plotting
ssp_avg_std["value"] = ssp_avg_std["mean_value"].astype(float)
ssp_avg_std["std"] = ssp_avg_std["std_value"].astype(float)
non_ssp_data = consolidated_results[~consolidated_results["Task"].str.contains("ssp")]
non_ssp_data["value"] = non_ssp_data["value"].astype(float)
non_ssp_data["std"] = np.nan  # No standard deviation for non-SSP tasks in this context

# Combine SSP average with non-SSP data for plotting
combined_data = pd.concat(
    [non_ssp_data, ssp_avg_std[["Task", "model", "value", "std"]]], ignore_index=True
)

# Plotting combined tasks
plt.figure(figsize=(14, 7))
ax = sns.barplot(data=combined_data, x="Task", y="value", hue="model", ci=None)

# Set plot title and labels
ax.set_ylabel("Metric Value")
ax.set_xlabel("Task")
ax.legend(title="Model", loc="upper left")


# Annotate each bar with the numeric value
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(
            format(p.get_height(), ".4f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
        )
        
# Compute and display the delta values
for i, task in enumerate(task_order):
    task_data = combined_data[combined_data["Task"] == task]
    models = task_data["model"].unique()
    if len(models) == 2:
        model_1_value = task_data[task_data["model"] == models[0]]["value"].values[0]
        model_2_value = task_data[task_data["model"] == models[1]]["value"].values[0]
        delta = model_1_value - model_2_value
        x = i  # Position in the plot
        height = max(model_1_value, model_2_value)
        plt.text(x, height + 0.04, f'Δ={delta:.2f}', ha="center", fontsize=12, color='red', fontweight='bold')

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("./plots/all_tasks_test.png", dpi=300, bbox_inches="tight")
plt.show()
