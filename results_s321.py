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

# Process data for fluorescence task
fluorescence_results = process_data(cv_data, "fluorescence", "eval_spearmanr")

# Setting up the visualization
sns.set_style("whitegrid")
sns.set_context("talk")

# Create a new figure
plt.figure(figsize=(10, 6))

# Positions of the bars
bar_positions = np.arange(len(fluorescence_results))


# Assuming 'cv_data' contains the relevant filtered data for the fluorescence task
fluorescence_data = cv_data[cv_data['task'] == 'fluorescence']

# Separate values for each model
ankh_values = fluorescence_data[fluorescence_data['model'] == 'ankh']['value'].values
p2s_values = fluorescence_data[fluorescence_data['model'] == 'p2s']['value'].values

# Perform paired t-test
_, p_value = ttest_rel(ankh_values, p2s_values)

# Add the p-value to the plot as a text annotation
plt.figure(figsize=(10, 6))

# Plotting each bar individually and annotating with p-value
for idx, row in fluorescence_results.iterrows():
    plt.bar(bar_positions[idx], row['mean'], yerr=row['std'], align='center', alpha=0.7, capsize=10, label=row['model'])

plt.xticks(bar_positions, fluorescence_results['model'])
plt.title('Fluorescence Prediction Performance')
plt.xlabel('Model')
plt.ylabel('Spearman\'s ρ (mean ± std)')

# Annotating the plot with the p-value
plt.text(0.5, max(fluorescence_results['mean']) + 0.05, f'p-value: {p_value:.3e}', ha='center', fontsize=12)

plt.legend()
plt.tight_layout()
plt.show()
