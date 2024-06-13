import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


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


# Function to plot the COG category distributions side by side
def plot_cog_distributions(cog_counts_correct, cog_counts_misclassified, task_name):
    # Combine and sort categories
    all_categories = list(
        set(cog_counts_correct.keys()) | set(cog_counts_misclassified.keys())
    )
    all_categories.sort()

    # Prepare counts for plotting
    correct_counts = [
        cog_counts_correct.get(category, 0) for category in all_categories
    ]
    misclassified_counts = [
        cog_counts_misclassified.get(category, 0) for category in all_categories
    ]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.35
    index = range(len(all_categories))

    bars1 = ax.bar(index, correct_counts, bar_width, label="Correctly Classified")
    bars2 = ax.bar(
        [p + bar_width for p in index],
        misclassified_counts,
        bar_width,
        label="Misclassified",
    )

    ax.set_xlabel("COG Category")
    ax.set_ylabel("Frequency")
    # ax.set_title(f'COG Category Distribution in {task_name} Sequences')
    ax.set_xticks([p + bar_width / 2 for p in index])
    ax.set_xticklabels(all_categories, rotation=45)
    ax.legend()

    # save the plot
    # plt.savefig(
    #     f"./eggNOG/{task_name}_COG_distribution.png", bbox_inches="tight", dpi=300
    # )
    plt.tight_layout()
    plt.show()


# Apply the code for each task
tasks = ["ionchannels", "localization", "mp", "solubility", "transporters"]

for task in tasks:
    correct_file = f"./eggNOG/{task}_correctly_classified.emapper.annotations.tsv"
    misclassified_file = f"./eggNOG/{task}_misclassified.emapper.annotations.tsv"

    cog_counts_correct = extract_cog_counts(correct_file)
    cog_counts_misclassified = extract_cog_counts(misclassified_file)

    # print the COG category counts for correct and misclassified sequences
    print(f"Correctly classified {task} sequences:")
    print(cog_counts_correct)
    print(f"Misclassified {task} sequences:")
    print(cog_counts_misclassified)

    plot_cog_distributions(cog_counts_correct, cog_counts_misclassified, task.capitalize())
