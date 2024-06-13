import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


# Function to extract and count COG categories from the 'COG_category' column
def extract_and_count_cogs(df):
    cog_counts = Counter()
    for cogs in df["COG_category"]:
        if isinstance(cogs, str):  # Only process the entry if it's a string
            # Splitting the COG categories and counting each occurrence
            for cog in cogs.split(","):
                cog_counts[cog] += 1
    return cog_counts


# Function to read the data and extract COG category counts
def extract_cog_counts(file_path):
    data = pd.read_csv(file_path, sep="\t", skiprows=4)  # Skipping the header rows
    cog_counts = extract_and_count_cogs(data)
    return cog_counts


# Function to create a combined DataFrame for all tasks
def create_combined_df(tasks, correct_dir, misclassified_dir):
    all_data = []
    for task in tasks:
        correct_file = (
            f"{correct_dir}/{task}_correctly_classified.emapper.annotations.tsv"
        )
        misclassified_file = (
            f"{misclassified_dir}/{task}_misclassified.emapper.annotations.tsv"
        )

        cog_counts_correct = extract_cog_counts(correct_file)
        cog_counts_misclassified = extract_cog_counts(misclassified_file)

        for category, count in cog_counts_correct.items():
            all_data.append(
                {
                    "Task": task,
                    "COG_Category": category,
                    "Count": count,
                    "Type": "Correctly Classified",
                }
            )
        for category, count in cog_counts_misclassified.items():
            all_data.append(
                {
                    "Task": task,
                    "COG_Category": category,
                    "Count": count,
                    "Type": "Misclassified",
                }
            )

    combined_df = pd.DataFrame(all_data)
    return combined_df


# Function to plot the top 10 most problematic COG categories as bar plot and heatmap
def plot_top_problematic_cog_distributions(df, top_n=10):
    # Filter for misclassified data and get the top N categories
    misclassified_df = df[df["Type"] == "Misclassified"]
    top_cog_categories = (
        misclassified_df.groupby("COG_Category")["Count"].sum().nlargest(top_n).index
    )

    # Filter the original dataframe to only include these top COG categories
    filtered_df = df[df["COG_Category"].isin(top_cog_categories)]

    # Bar plot
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=filtered_df,
        x="COG_Category",
        y="Count",
        hue="Type",
        estimator=sum,
        ci=None,
    )
    # plt.title(f"Top {top_n} Most Problematic COG Categories")
    plt.xlabel("COG Category")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Type")
    plt.tight_layout()
    # plt.show()
    plt.savefig("./eggNOG/top_cog_categories_bar.png", dpi=300, bbox_inches="tight")

    # Heatmap
    pivot_df = filtered_df.pivot_table(
        index="COG_Category",
        columns=["Task", "Type"],
        values="Count",
        aggfunc="sum",
        fill_value=0,
    )
    plt.figure(figsize=(16, 10))
    sns.heatmap(pivot_df, annot=True, fmt="d", cmap="YlGnBu")
    # plt.title(f"Heatmap of Top {top_n} Most Problematic COG Categories")
    plt.xlabel("Task and Classification Type")
    plt.ylabel("COG Category")
    plt.tight_layout()
    # plt.show()
    plt.savefig("./eggNOG/top_cog_categories_heatmap.png", dpi=300, bbox_inches="tight")

    # Print the table
    summary_table = filtered_df.pivot_table(
        index=["COG_Category"],
        columns="Type",
        values="Count",
        aggfunc="sum",
        fill_value=0,
    )
    print(summary_table)
    summary_table.to_csv("./eggNOG/summary_table_top10.csv")


# Function to create a summary table
def create_summary_table(df):
    summary_table = df.pivot_table(
        index=["Task", "COG_Category"],
        columns="Type",
        values="Count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    return summary_table


# Define tasks and directories
tasks = ["ionchannels", "localization", "mp", "solubility", "transporters"]
correct_dir = "./eggNOG"
misclassified_dir = "./eggNOG"

# Create the combined DataFrame and plot
combined_df = create_combined_df(tasks, correct_dir, misclassified_dir)
summary_table = create_summary_table(combined_df)

# Display the summary table
print(summary_table)

# Plot the top 10 most problematic COG categories as bar plot and heatmap
plot_top_problematic_cog_distributions(combined_df, top_n=10)
