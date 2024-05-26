import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from settings.settings import task_order
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
sort_order = ["cb513", "ts115", "casp12", "casp13", "casp14"]

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

consolidated_results["Metric"] = consolidated_results["Task"].apply(
    lambda x: metric_format_map[tasks_metrics.get(x.split("_")[0], "")]
)


# Format values: round and convert to percentage if necessary
def format_values(row):
    if row["Metric"] == "Accuracy":
        return f"{row['value'] * 100:.2f}%"
    else:
        return f"{row['value']:.4f}"


consolidated_results["value"] = consolidated_results.apply(format_values, axis=1)

consolidated_results["Task"] = consolidated_results["Task"].replace(
    {"SSP_ssp3": "ssp3", "SSP_ssp8": "ssp8"}
)
consolidated_results["Task"] = pd.Categorical(
    consolidated_results["Task"], categories=task_order, ordered=True
)

# more detail results:

# Assuming consolidated_results is prepared and loaded

# Filter to include only SSP tasks and convert percentage strings to float values
ssp_data = consolidated_results[consolidated_results["Task"].str.contains("ssp")]
ssp_data["value"] = ssp_data["value"].replace("%", "", regex=True).astype(float) / 100.0

# Convert categorical data to string if necessary and create a new 'Detailed Task' column
ssp_data["Detailed Task"] = ssp_data["Task"].astype(str) + " " + ssp_data["data_split"]

# Separate plots for SSP3 and SSP8
for ssp_type in ["ssp3", "ssp8"]:
    # Filter data for the current SSP type
    plot_data = ssp_data[ssp_data["Task"].str.contains(ssp_type)]

    # Set up the plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=plot_data,
        x="Detailed Task",
        y="value",
        hue="model",
    )

    # Set plot title and labels
    # ax.set_title(f'Comparison of Models Across {ssp_type.upper()} Tasks and Datasets')
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Task and Dataset")
    ax.legend(title="Model", loc="upper left")

    # Set y-axis limits based on data
    ax.set_ylim(0, plot_data["value"].max() * 1.1)

    # Annotate each bar with the numeric value
    for p in ax.patches:
        if p.get_height() > 0:  # Only annotate bars with a non-zero height
            ax.annotate(
                format(p.get_height(), ".2f"),
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 9),
                textcoords="offset points",
            )

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        f"./plots/{ssp_type}_tasks_test.png", dpi=300, bbox_inches="tight"
    )  # Save plot
    # plt.show()  # Optionally display the plot


#  ------------

# Sort by data_split within each task group for SSP tasks
ssp_tasks = ["ssp3", "ssp8"]
consolidated_results.loc[consolidated_results["Task"].isin(ssp_tasks), "data_split"] = (
    pd.Categorical(
        consolidated_results.loc[
            consolidated_results["Task"].isin(ssp_tasks), "data_split"
        ],
        categories=sort_order,
        ordered=True,
    )
)
consolidated_results.sort_values(["Task", "data_split"], inplace=True)

# Filter to include only SSP tasks
ssp_data = consolidated_results[
    consolidated_results["Task"].str.contains("ssp", case=False)
]

# Convert percentage strings to float values and handle other values already in float
ssp_data["value"] = ssp_data["value"].replace("%", "", regex=True).astype(float) / 100.0

# Remove unused categories from the Task column
ssp_data["Task"] = ssp_data["Task"].cat.remove_unused_categories()

# Plotting setup for SSP tasks
plt.figure(figsize=(14, 7))
ax = sns.barplot(
    data=ssp_data,
    x="Task",
    y="value",
    hue="model",
    order=ssp_data["Task"].cat.categories,
)

# ax.set_title('Comparison of Models Across SSP Tasks and Datasets')
ax.set_ylabel("Metric Value")
ax.set_xlabel("Task")
ax.legend(title="Model", loc="upper left")

# Correctly setting y-axis limits based on the data
ax.set_ylim(0, ssp_data["value"].max() * 1.1)

# Annotate each bar with the numeric value
for p in ax.patches:
    if p.get_height() > 0:  # Only annotate bars with a non-zero height
        ax.annotate(
            format(p.get_height(), ".2f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
        )

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("./plots/ssp_tasks_test.png", dpi=300, bbox_inches="tight")  # Save plot
# plt.show()  # Show plot

# Filtering out SSP tasks and removing unused categories
non_ssp_data = consolidated_results[
    ~consolidated_results["Task"].str.contains("ssp", case=False)
]
non_ssp_data["Task"] = non_ssp_data["Task"].cat.remove_unused_categories()

# Ensure the 'value' column is treated as float
non_ssp_data["value"] = non_ssp_data["value"].astype(float)

# Plotting setup for non-SSP tasks
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    data=non_ssp_data,
    x="Task",
    y="value",
    hue="model",
    order=non_ssp_data["Task"].cat.categories,
)

# ax.set_title('Comparison of Models Across Non-SSP Tasks')
ax.set_ylabel("Metric Value")
ax.set_xlabel("Task")
ax.legend(title="Model", loc="upper left")

# Correctly setting y-axis limits based on the data
ax.set_ylim(0, non_ssp_data["value"].max() * 1.1)

# Annotate each bar with the numeric value
for p in ax.patches:
    if p.get_height() > 0:  # Only annotate bars with a non-zero height
        ax.annotate(
            format(p.get_height(), ".2f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
        )

plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()
plt.savefig("./plots/non_ssp_tasks_test.png", dpi=300, bbox_inches="tight")

print(consolidated_results)

print(
    consolidated_results.to_latex(
        index=False,
        column_format="lcccc",  # Define the column alignment
        header=True,  # Include the headers
        longtable=False,  # Set to True if the table is expected to span multiple pages
        caption="Summary of Model Results across Various Tasks",  # Add a caption
        label="tab:model_results",  # Add a label for referencing
        escape=True,  # Disable escaping LaTeX characters
    )
)

# Assuming 'df' is your DataFrame loaded with the consolidated data
# Splitting the data into non-SSP and SSP datasets
non_ssp_data = consolidated_results[
    consolidated_results["Task"].isin(
        [
            "fluorescence",
            "solubility",
            "localization",
            "ionchannels",
            "transporters",
            "mp",
        ]
    )
]
ssp_data = consolidated_results[consolidated_results["Task"].str.contains("ssp")]

# Plotting Non-SSP Tasks
plt.figure(figsize=(10, 6))
sns.barplot(data=non_ssp_data, x="Task", y="value", hue="model", ci=None)
plt.title("Performance Comparison across Non-SSP Tasks")
plt.ylabel("Metric Value")
plt.xticks(rotation=45)
plt.legend(title="Model")
plt.tight_layout()
plt.show()

# Plotting SSP Tasks
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=ssp_data,
    x="data_split",
    y="value",
    hue="model",
    style="Task",
    markers=True,
    dashes=False,
)
plt.title("Model Performance on SSP Tasks across Datasets")
plt.ylabel("Accuracy (%)")
plt.xlabel("Dataset")
plt.xticks(rotation=45)
plt.legend(title="Model and SSP Type")
plt.tight_layout()
plt.show()
