from Bio import SeqIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Define tasks and their respective number of alignment files
tasks = {
    "transporters": 5,
    "localization": 5,
    "solubility": 5,
    "ionchannels": 5,
    "mp": 5,
}


# Function to parse the FASTA file and return sequences
def parse_fasta(fasta_file_path):
    with open(fasta_file_path, "r") as handle:
        fasta_records = list(SeqIO.parse(handle, "fasta"))
    return fasta_records


# Function to calculate sequence identity
def sequence_identity(seq1, seq2):
    matches = sum(res1 == res2 for res1, res2 in zip(seq1, seq2))
    length = max(len(seq1), len(seq2))  # Use the total length of the alignment
    return matches / length if length > 0 else 0


# Dictionary to store the average identity for each task
task_avg_identities = {}

# Process each task and its alignments
for task, num_files in tasks.items():
    task_identities = []

    for i in range(num_files):
        # Construct the file path
        fasta_file_path = f"./task_alignments/{task}/{task}_msa_{i}.fasta"

        # Parse the FASTA file
        sequences = parse_fasta(fasta_file_path)

        # Separate misclassified and correctly classified sequences
        misclassified_seqs = [
            str(record.seq) for record in sequences if "misclassified" in record.id
        ]
        correctly_classified_seqs = [
            str(record.seq)
            for record in sequences
            if "correctly_classified" in record.id
        ]

        # Calculate pairwise sequence identity for each misclassified to correctly classified sequence
        identities = [
            sequence_identity(misclassified_seqs[0], seq)
            for seq in correctly_classified_seqs
        ]

        # Calculate the average identity for this file
        avg_identity = sum(identities) / len(identities) if identities else 0
        task_identities.append(avg_identity)

        print(f"Task: {task}, Alignment {i}, Average Identity: {avg_identity}")

    # Calculate and store the average identity for the task
    task_avg_identities[task] = sum(task_identities) / num_files

# Create a DataFrame for the results
df = pd.DataFrame(
    list(task_avg_identities.items()), columns=["Task", "Average Identity"]
)

# Display the average identity for each task
print(df)

# Calculate and display the global average identity
global_avg_identity = df["Average Identity"].mean()
print(f"Global Average Identity: {global_avg_identity}")

# Plot the results
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
sns.set_palette("colorblind")
plt.bar(df["Task"], df["Average Identity"], color="skyblue")
plt.xlabel("Task")
plt.ylabel("Average Sequence Identity")
# plt.title(
#     "Average Sequence Identity for Misclassified vs Correctly Classified Sequences"
# )
plt.axhline(
    y=global_avg_identity,
    color="r",
    linestyle="--",
    label=f"Global Average Identity: {global_avg_identity:.2f}",
)
plt.legend()
# plt.show()
plt.savefig("./task_alignments/average_identity_plot.png", dpi=300, bbox_inches="tight")
