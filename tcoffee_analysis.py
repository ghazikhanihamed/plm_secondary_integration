import os
from Bio import AlignIO
import matplotlib.pyplot as plt
import numpy as np


def calculate_conservation(msa):
    """Calculate conservation scores for each position in the MSA."""
    # Placeholder function; implement based on your chosen metric
    return np.random.rand(msa.get_alignment_length())  # Example with random data


def plot_conservation(task, msa_conserved, msa_misclassified, output_folder):
    # Determine the maximum length of the two arrays
    max_len = max(len(msa_conserved), len(msa_misclassified))

    # Ensure both arrays are the same length by padding the shorter array
    if len(msa_conserved) < max_len:
        msa_conserved = np.pad(
            msa_conserved, (0, max_len - len(msa_conserved)), "constant"
        )
    if len(msa_misclassified) < max_len:
        msa_misclassified = np.pad(
            msa_misclassified, (0, max_len - len(msa_misclassified)), "constant"
        )

    plt.figure(figsize=(12, 6))
    positions = np.arange(max_len)
    plt.plot(positions, msa_conserved, label="Correctly Classified", color="green")
    plt.plot(positions, msa_misclassified, label="Misclassified", color="red")
    # plt.title(f'Conservation Scores for {task}')
    plt.xlabel("Alignment Position")
    plt.ylabel("Conservation Score")
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"{task}_conservation_plot.png"))
    plt.close()


def analyze_task(task, output_folder):
    """Analyze MSA for a given task and create conservation plots."""
    msa_file_conserved = os.path.join(
        output_folder, task, f"{task}_msa_0.fasta"
    )  # Adjust as needed
    msa_file_misclassified = os.path.join(
        output_folder, task, f"{task}_msa_1.fasta"
    )  # Adjust as needed

    # Parse the MSA output files
    msa_conserved = AlignIO.read(msa_file_conserved, "fasta")
    msa_misclassified = AlignIO.read(msa_file_misclassified, "fasta")

    # Calculate conservation scores
    conserved_scores = calculate_conservation(msa_conserved)
    misclassified_scores = calculate_conservation(msa_misclassified)

    # # print the conservation scores for task and misclassified and conserved
    # print(f"Conservation scores for task {task}: {conserved_scores}")
    # print(f"Conservation scores for misclassified {task}: {misclassified_scores}")

    # Generate conservation plot
    plot_conservation(task, conserved_scores, misclassified_scores, output_folder)


# Example usage
output_base_folder = "./task_alignments"
tasks = ["transporters", "localization", "solubility", "ionchannels", "mp"]

for task in tasks:
    analyze_task(task, output_base_folder)
