import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the provided code snippet and data manipulation as described
# The data loading part is not executable in this environment, so we'll simulate relevant data

# Load the uploaded CSV file to inspect its contents
file_path = "./results/combined_results.csv"
data = pd.read_csv(file_path)

# Filter data for cross-validation results
cv_data = data[data["data_split"] == "cross-validation"]


# Function to filter and process data based on the task and required metric
def process_data(df, task, metric, ssp=None):
    if ssp:  # If ssp is specified, further filter by ssp column
        df_filtered = df[
            (df["task"] == task) & (df["metric"] == metric) & (df["ssp"] == ssp)
        ]
    else:
        df_filtered = df[(df["task"] == task) & (df["metric"] == metric)]

    # Make sure to create a copy of the DataFrame to avoid SettingWithCopyWarning
    df_filtered = df_filtered.copy()

    # Replace model codes with full names, using a copy of the DataFrame to avoid warnings
    df_filtered["model"] = df_filtered["model"].replace(
        {"ankh": "Ankh", "p2s": "TooT-PLM-P2S"}
    )

    # Calculate mean and std for each model within the filtered data
    results = df_filtered.groupby("model")["value"].agg(["mean", "std"]).reset_index()
    return results


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

    if len(ankh_values) == len(p2s_values) and len(ankh_values) > 1:
        _, p_value = ttest_rel(ankh_values, p2s_values)
        # Format the p-value with one decimal in scientific notation
        p_value[metric.replace("eval_", "")] = np.format_float_scientific(
            p_value, precision=1
        )
    else:
        p_value = np.nan
        p_value[metric.replace("eval_", "")] = np.format_float_scientific(
            p_value, precision=1
        )  # Handle NaN case

    # # Ensure there is a pair of results for each fold before performing the test
    # if len(ankh_values) == len(p2s_values) and len(ankh_values) > 1:
    #     try:
    #         # Use paired t-test
    #         _, p_value = ttest_rel(ankh_values, p2s_values)
    #     except ValueError:
    #         # In case of an error (e.g., all values are identical), set p-value to NaN
    #         p_value = np.nan
    # else:
    #     # If there's not a matching number of results for each model, we can't compute the p-value
    #     p_value = np.nan

    return p_value


# Process data for fluorescence task
fluorescence_results = process_data(cv_data, "fluorescence", "eval_spearmanr")

# Processing data for the "Fluorescence Prediction" task
fluorescence_table = process_data(cv_data, "fluorescence", "eval_spearmanr")

# Compute p-value for the "Fluorescence Prediction" task
p_value_fluorescence = compute_p_value(cv_data, "fluorescence", "eval_spearmanr")

# Add the computed p-value to the fluorescence results DataFrame and convert it to scientific notation
fluorescence_table["P-Value"] = p_value_fluorescence
fluorescence_table["P-Value"] = fluorescence_table["P-Value"].apply(
    lambda x: np.format_float_scientific(x, precision=1)
)

# Filter data for Ankh and TooT-PLM-P2S models
ankh_fluorescence_data = cv_data[
    (cv_data["task"] == "fluorescence") & (cv_data["model"] == "ankh")
]
p2s_fluorescence_data = cv_data[
    (cv_data["task"] == "fluorescence") & (cv_data["model"] == "p2s")
]

# Calculate max, min, and median for Ankh model
fluorescence_table.loc[fluorescence_table["model"] == "Ankh", "Max"] = (
    ankh_fluorescence_data["value"].max()
)
fluorescence_table.loc[fluorescence_table["model"] == "Ankh", "Min"] = (
    ankh_fluorescence_data["value"].min()
)
fluorescence_table.loc[fluorescence_table["model"] == "Ankh", "Median"] = (
    ankh_fluorescence_data["value"].median()
)

# Calculate max, min, and median for TooT-PLM-P2S model
fluorescence_table.loc[fluorescence_table["model"] == "TooT-PLM-P2S", "Max"] = (
    p2s_fluorescence_data["value"].max()
)
fluorescence_table.loc[fluorescence_table["model"] == "TooT-PLM-P2S", "Min"] = (
    p2s_fluorescence_data["value"].min()
)
fluorescence_table.loc[fluorescence_table["model"] == "TooT-PLM-P2S", "Median"] = (
    p2s_fluorescence_data["value"].median()
)

fluorescence_table["Mean ± Std"] = fluorescence_table.apply(
    lambda row: f"{row['mean']:.3f} ± {row['std']:.3f}", axis=1
)

# Updating the table for LaTeX with correct max, min, and median values for each model
fluorescence_latex_table = fluorescence_table[
    ["model", "Mean ± Std", "Max", "Min", "Median", "P-Value"]
].to_latex(
    index=False,
    caption="Corrected Performance Metrics for Fluorescence Prediction Task (Spearman's ρ) Separately for Each Model",
    label="tab:corrected_fluorescence_prediction",
    column_format="lccccc",
    float_format="%.3f",
    escape=False,
)


print(fluorescence_latex_table)

# Setting up the visualization
sns.set_style("whitegrid")
sns.set_context("talk")

# Create a new figure
plt.figure(figsize=(10, 6))

# Positions of the bars
bar_positions = np.arange(len(fluorescence_results))


# Assuming 'cv_data' contains the relevant filtered data for the fluorescence task
fluorescence_data = cv_data[cv_data["task"] == "fluorescence"]

# Separate values for each model
ankh_values = fluorescence_data[fluorescence_data["model"] == "ankh"]["value"].values
p2s_values = fluorescence_data[fluorescence_data["model"] == "p2s"]["value"].values

# Perform paired t-test
_, p_value = ttest_rel(ankh_values, p2s_values)

# Add the p-value to the plot as a text annotation
plt.figure(figsize=(10, 6))

# Plotting each bar individually and annotating with p-value
for idx, row in fluorescence_results.iterrows():
    plt.bar(
        bar_positions[idx],
        row["mean"],
        yerr=row["std"],
        align="center",
        alpha=0.7,
        capsize=10,
        label=row["model"],
    )

plt.xticks(bar_positions, fluorescence_results["model"])
# plt.title('Fluorescence Prediction Performance')
plt.xlabel("Model")
plt.ylabel("Spearman's ρ (mean ± std)")

# Annotating the plot with the p-value
plt.text(
    0.5,
    max(fluorescence_results["mean"]) + 0.02,
    f"p-value: {p_value:.3e}",
    ha="center",
    fontsize=14,
)

plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("./plots/plot3.2.1_flu.png", bbox_inches="tight", dpi=300)
plt.close()
