import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Function to extract motif information from the text file
def extract_motif_info_from_txt(file_path):
    motifs = {}
    with open(file_path, "r") as file:
        lines = file.readlines()
        motif_id = None
        for line in lines:
            if line.startswith("MOTIF"):
                motif_id = line.split()[1]
                motifs[motif_id] = {"occurrences": 0}
            elif motif_id and line.startswith(" "):
                motifs[motif_id]["occurrences"] += 1
    return motifs


# List of tasks
tasks = ["solubility", "localization", "ionchannels", "transporters", "mp"]

# Base path where the meme files are located
base_path = "./meme_files/"

# Initialize an empty DataFrame for the box plot data
box_plot_data = pd.DataFrame(columns=["Task", "Classification", "Occurrences"])

# Initialize counters for the number of sequences and total motifs
total_misclassified_sequences = 0
total_correctly_classified_sequences = 0
total_motifs = 0

# Store motif counts for each task
motif_counts = {}

# Store a few example motifs for display
example_motifs = []

# Loop through each task
for task in tasks:
    # Construct file paths
    misclassified_txt_path = f"{base_path}/{task}_common_misclassified_meme.txt"
    correctly_classified_txt_path = (
        f"{base_path}/{task}_common_correctly_classified_meme.txt"
    )

    # Extract motifs
    misclassified_motifs = extract_motif_info_from_txt(misclassified_txt_path)
    correctly_classified_motifs = extract_motif_info_from_txt(
        correctly_classified_txt_path
    )

    # Count the number of sequences
    total_misclassified_sequences += len(misclassified_motifs)
    total_correctly_classified_sequences += len(correctly_classified_motifs)
    total_motifs += len(misclassified_motifs) + len(correctly_classified_motifs)

    # Store motif counts
    motif_counts[task] = {
        "misclassified": len(misclassified_motifs),
        "correctly_classified": len(correctly_classified_motifs),
    }

    # Append the motif occurrences to the DataFrame
    for motif, info in misclassified_motifs.items():
        box_plot_data = box_plot_data._append(
            {
                "Task": task,
                "Classification": "Misclassified",
                "Occurrences": info["occurrences"],
            },
            ignore_index=True,
        )
    for motif, info in correctly_classified_motifs.items():
        box_plot_data = box_plot_data._append(
            {
                "Task": task,
                "Classification": "Correctly Classified",
                "Occurrences": info["occurrences"],
            },
            ignore_index=True,
        )
        
    # Store example motifs for display
    if len(example_motifs) < 5:
        example_motifs.append((task, list(misclassified_motifs.keys())[:3], list(correctly_classified_motifs.keys())[:3]))

# Print the summary statistics
print(f"Total misclassified sequences: {total_misclassified_sequences}")
print(f"Total correctly classified sequences: {total_correctly_classified_sequences}")
print(f"Total motifs found: {total_motifs}")

# Print example motifs
print("\nExample Motifs:")
for task, misclassified_motif, correctly_classified_motif in example_motifs:
    print(f"Task: {task}, Misclassified Motif: {misclassified_motif}, Correctly Classified Motif: {correctly_classified_motif}")

# Print motif counts for each task
print("\nMotif Counts:")
for task, counts in motif_counts.items():
    print(f"Task: {task}, Misclassified Motifs: {counts['misclassified']}, Correctly Classified Motifs: {counts['correctly_classified']}")

# Plotting the box plot
plt.figure(figsize=(14, 8))
sns.boxplot(
    x="Task", y="Occurrences", hue="Classification", data=box_plot_data, palette="Set2"
)
plt.xlabel("Tasks")
plt.ylabel("Occurrences of Motifs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# plt.savefig("./meme_files/motif_occurrences_boxplot.png", dpi=300, bbox_inches="tight")




# # Initialize an empty DataFrame for the heatmap data
# heatmap_data = pd.DataFrame()

# # Loop through each task
# for task in tasks:
#     # Construct file paths
#     misclassified_txt_path = f'{base_path}/{task}_common_misclassified_meme.txt'
#     correctly_classified_txt_path = f'{base_path}/{task}_common_correctly_classified_meme.txt'

#     # Extract motifs
#     misclassified_motifs = extract_motif_info_from_txt(misclassified_txt_path)
#     correctly_classified_motifs = extract_motif_info_from_txt(correctly_classified_txt_path)

#     # Merge the motif data
#     for motif, info in misclassified_motifs.items():
#         heatmap_data.loc[motif, f'{task}_misclassified'] = info['occurrences']
#     for motif, info in correctly_classified_motifs.items():
#         heatmap_data.loc[motif, f'{task}_correctly_classified'] = info['occurrences']

# # Replace NaN values with 0
# heatmap_data.fillna(0, inplace=True)

# # Log transform the data to better visualize the range of occurrences
# heatmap_data_log = np.log1p(heatmap_data)

# # Plotting the heatmap
# plt.figure(figsize=(14, 10))
# sns.heatmap(heatmap_data_log, annot=True, fmt=".1f", cmap='viridis', linewidths=.5)
# plt.title('Log-Scaled Occurrences of Motifs in Misclassified vs. Correctly Classified Sequences')
# plt.ylabel('Motifs')
# plt.xlabel('Tasks and Classification')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()

# # Show the plot
# plt.show()

# # Loop through each task
# for task in tasks:
#     # Construct file paths
#     misclassified_txt_path = f'{base_path}/{task}_common_misclassified_meme.txt'
#     correctly_classified_txt_path = f'{base_path}/{task}_common_correctly_classified_meme.txt'

#     # Extract motifs
#     misclassified_motifs_info = extract_motif_info_from_txt(misclassified_txt_path)
#     correctly_classified_motifs_info = extract_motif_info_from_txt(correctly_classified_txt_path)

#     # Prepare data for plotting
#     motifs = list(set(misclassified_motifs_info.keys()) | set(correctly_classified_motifs_info.keys()))
#     misclassified_counts = [misclassified_motifs_info.get(motif, {'occurrences': 0})['occurrences'] for motif in motifs]
#     correctly_classified_counts = [correctly_classified_motifs_info.get(motif, {'occurrences': 0})['occurrences'] for motif in motifs]

#     # Plotting
#     x = np.arange(len(motifs))  # the label locations
#     width = 0.35  # the width of the bars

#     fig, ax = plt.subplots(figsize=(12, 6))
#     rects1 = ax.bar(x - width/2, misclassified_counts, width, label='Misclassified', color='tomato')
#     rects2 = ax.bar(x + width/2, correctly_classified_counts, width, label='Correctly Classified', color='lightgreen')

#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     ax.set_ylabel('Occurrences')
#     ax.set_title(f'Comparison of Motif Occurrences in {task.capitalize()} Misclassified vs. Correctly Classified Sequences')
#     ax.set_xticks(x)
#     ax.set_xticklabels(motifs, rotation=45)
#     ax.legend()

#     ax.bar_label(rects1, padding=3)
#     ax.bar_label(rects2, padding=3)

#     plt.tight_layout()

#     # Show the plot
#     plt.show()
