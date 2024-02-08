import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
from settings.settings import task_order

# Load the uploaded CSV file to inspect its contents
file_path = "./results/combined_results.csv"
data = pd.read_csv(file_path)

# Filter data for cross-validation results
cv_data = data[data["data_split"] == "cross-validation"]


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


# Define tasks and their corresponding metrics
tasks_metrics = {
    "fluorescence": "eval_spearmanr",
    "SSP": "eval_accuracy",  # For ssp, we will handle ssp3 and ssp8 separately
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
            summary_results[f"{ssp_type}"] = process_data(
                cv_data, task, metric, ssp_type
            )
    else:
        summary_results[task] = process_data(cv_data, task, metric)

# Map the metric names to their desired format for display
metric_format_map = {
    "eval_spearmanr": "Spearman's ρ",
    "eval_mcc": "MCC",
    "eval_accuracy": "Accuracy",
}


# Function to format the task name and assign the corresponding metric
def assign_metric(task):
    if task == "fluorescence":
        return metric_format_map["eval_spearmanr"]
    elif task.startswith("ssp"):
        return metric_format_map["eval_accuracy"]
    else:
        return metric_format_map["eval_mcc"]


# Compute p-values for each task and add them to the summary_results dictionary
for task, results in summary_results.items():
    if task.startswith("ssp"):
        # Extract 'ssp3' or 'ssp8'
        ssp_type = task
        metric = "eval_accuracy"
        p_value = compute_p_value(cv_data, "SSP", metric, ssp_type)
    else:
        metric = tasks_metrics[task]
        p_value = compute_p_value(cv_data, task, metric)

    summary_results[task]["P-Value"] = p_value


# Update the consolidated_results DataFrame to include the P-Value
consolidated_results = pd.DataFrame()
for task, results in summary_results.items():
    results["Task"] = task
    consolidated_results = pd.concat([consolidated_results, results], ignore_index=True)

# # Reorder and rename columns for clarity in the LaTeX table
consolidated_results = consolidated_results[["Task", "model", "mean", "std", "P-Value"]]
consolidated_results.columns = ["Task", "Model", "Mean", "Std", "P-Value"]

# Combine mean and standard deviation into one column with the format "mean ± std"
consolidated_results["Mean ± Std"] = consolidated_results.apply(
    lambda row: f"{row['Mean']:.3f} ± {row['Std']:.3f}", axis=1
)

# Apply the function to create a new 'Metric' column
consolidated_results["Metric"] = consolidated_results["Task"].apply(assign_metric)

# Duplicate P-Value for each model within the same task
consolidated_results["P-Value"] = consolidated_results.groupby("Task")[
    "P-Value"
].transform("first")

# we convert p-values to scientific notation
consolidated_results["P-Value"] = consolidated_results["P-Value"].apply(
    lambda x: np.format_float_scientific(x, precision=1)
)

# Update the DataFrame to reorder columns including the new 'Metric' column
final_summary = consolidated_results[
    ["Task", "Model", "Metric", "Mean ± Std", "P-Value"]
]

final_summary["Task"] = pd.Categorical(
    consolidated_results["Task"], categories=task_order, ordered=True
)

final_summary = final_summary.sort_values(["Task"])

# Generate the updated LaTeX table code with the 'Metric' column
final_latex_table = final_summary.to_latex(
    index=False,
    caption="Comparative Performance Overview of Ankh and TooT-PLM-P2S across Various Tasks",
    label="tab:performance_comparison",
    escape=False,
)
print(final_latex_table)


# Ensure 'Mean' and 'Std' columns exist
consolidated_results["Mean"] = consolidated_results["Mean ± Std"].apply(
    lambda x: float(x.split(" ± ")[0])
)
consolidated_results["Std"] = consolidated_results["Mean ± Std"].apply(
    lambda x: float(x.split(" ± ")[1])
)

num_models = len(consolidated_results["Model"].unique())

# Set the style and context with a smaller font scale
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
sns.set_palette("colorblind")

# Determine the number of unique models and tasks
num_models = len(consolidated_results['Model'].unique())
num_tasks = len(task_order)

# Calculate the width of the bars and the total group width
bar_width = 0.35
group_width = bar_width * num_models

# Create a figure and a set of subplots
plt.figure(figsize=(12, 6))

# Store the means and std devs for delta computation
means = {}
stds = {}

model_colors = {
    'Ankh': 'skyblue',     # Light blue color for good contrast with black
    'TooT-PLM-P2S': 'sandybrown'  # Light orange/brown color
}

# Iterate over the tasks and models to create the grouped bar plot
for i, task in enumerate(task_order):
    # Filter the data for the current task
    task_data = consolidated_results[consolidated_results['Task'] == task]
    
    # Iterate over the models for the current task
    for j, model in enumerate(task_data['Model'].unique()):
        # Get the mean and standard deviation for the current task and model
        mean = task_data[task_data['Model'] == model]['Mean'].values[0]
        std = task_data[task_data['Model'] == model]['Std'].values[0]
        
        # Store the means and std devs for delta computation
        if task not in means:
            means[task] = []
            stds[task] = []
        means[task].append(mean)
        stds[task].append(std)
        
        # Calculate the x position for the current bar
        x = i * (group_width + bar_width) + (j * bar_width)

        color = model_colors.get(model, 'grey')
        
        # Plot the bar with error bar
        plt.bar(x, mean, bar_width, color=color, edgecolor='white', yerr=std, capsize=3, label=model if i == 0 else "")
        plt.errorbar(x, mean, yerr=std, fmt='none', c='black', capsize=3, capthick=2)
        plt.text(x, mean - 0.1, f'{mean:.2f}', ha='center', va='bottom')

# Compute and display the delta values
for i, task in enumerate(task_order):
    # Compute the delta value (difference between means of the two models)
    if len(means[task]) == 2:
        delta = means[task][0] - means[task][1]
        # Calculate the position for the delta annotation in the middle of the group
        x = i * (group_width + bar_width) + bar_width / 2
        # Get the maximum height of the bars for the current task
        height = max(means[task][0], means[task][1])
        plt.text(x, height + 0.04, f'Δ={delta:.2f}', ha="center", fontsize=12, color='red', fontweight='bold')

# Customizing the plot
# plt.title("Comparative Performance Overview of Ankh and TooT-PLM-P2S")
plt.xlabel("Task")
plt.ylabel("Mean ± Std")
plt.xticks(np.arange(num_tasks) * (group_width + bar_width), task_order, rotation=45)

# Adjust the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), title="Model")

# Show the plot
plt.tight_layout()
# plt.show()

plt.savefig("./plots/plot3.1_model_comparison.png", bbox_inches='tight', dpi=300)
plt.close()