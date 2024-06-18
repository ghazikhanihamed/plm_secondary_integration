import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Function to extract motif information from the text file
def extract_motif_info_from_txt(file_path):
    motifs = {}

    try:
        with open(file_path, "r") as file:
            for line in file:
                if line.startswith("MOTIF"):
                    parts = line.split()
                    motif_id = parts[1]
                    sites_index = parts.index("sites") + 2  # The number of sites is two indices after "sites"
                    occurrences = int(parts[sites_index])
                    motifs[motif_id] = {"occurrences": occurrences}
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        # If the file is not found, return an empty dictionary
        return motifs

    return motifs

# List of tasks
tasks = ["solubility", "localization", "ionchannels", "transporters", "mp"]

# Base path where the meme files are located
base_path = "./meme_files/"

# Initialize an empty list for the plot data
plot_data = []

# Loop through each task
for task in tasks:
    # Construct file paths
    misclassified_txt_path = f"{base_path}/{task}_common_misclassified_meme.txt"
    correctly_classified_txt_path = f"{base_path}/{task}_common_correctly_classified_meme.txt"

    # Extract motifs
    misclassified_motifs = extract_motif_info_from_txt(misclassified_txt_path)
    correctly_classified_motifs = extract_motif_info_from_txt(correctly_classified_txt_path)

    # Append the motif occurrences to the list for misclassified and correctly classified motifs
    for motif, info in misclassified_motifs.items():
        plot_data.append({
            "Task": task,
            "Motif": motif,
            "Classification": "Misclassified",
            "Occurrences": info["occurrences"],
        })
    
    for motif, info in correctly_classified_motifs.items():
        plot_data.append({
            "Task": task,
            "Motif": motif,
            "Classification": "Correctly Classified",
            "Occurrences": info["occurrences"],
        })

# Convert the list to a DataFrame
plot_data_df = pd.DataFrame(plot_data)

# make latex table
print(plot_data_df.to_latex(index=False))

# Display the DataFrame as a table
print(plot_data_df)

# Create a bar plot
plt.figure(figsize=(14, 7))
sns.barplot(x="Task", y="Occurrences", hue="Classification", data=plot_data_df, palette="Set3")
plt.title("Motif Occurrences in Correctly and Misclassified Motifs Across Tasks")
plt.ylabel("Number of Occurrences")
plt.xlabel("Task")
plt.xticks(rotation=45)
plt.legend(title="Classification")
plt.tight_layout()
plt.show()
