import os
import subprocess
import random


def perform_msa(task, misclassified_seq, correctly_classified_seqs, output_path, i):
    # Create a combined file for MSA
    combined_file = os.path.join(output_path, f"{task}_combined_temp_{i}.fasta")
    with open(combined_file, "w") as outfile:
        # Write the misclassified sequence
        outfile.write(f">misclassified_{i}\n{misclassified_seq}\n")

        # Write the correctly classified sequences
        for j, seq in enumerate(correctly_classified_seqs):
            outfile.write(f">correctly_classified_{j}\n{seq}\n")

    # Run T-Coffee for MSA
    msa_output_file = os.path.join(output_path, f"{task}_msa_{i}.fasta")
    command = f"t_coffee -in {combined_file} -outfile {msa_output_file} -output fasta"
    subprocess.run(command, shell=True)

    # Remove the temporary combined file
    os.remove(combined_file)


def run_t_coffee_for_tasks(
    tasks, misclassified_folder, correctly_classified_folder, output_base_folder
):
    # Make sure the base output directory exists
    os.makedirs(output_base_folder, exist_ok=True)

    for task in tasks:
        output_folder = os.path.join(output_base_folder, task)
        os.makedirs(output_folder, exist_ok=True)

        misclassified_file = os.path.join(
            misclassified_folder, f"{task}_common_misclassified.fasta"
        )
        correctly_classified_file = os.path.join(
            correctly_classified_folder, f"{task}_common_correctly_classified.fasta"
        )

        # Read sequences from files and randomly select five from each
        misclassified_seqs = read_sequences(misclassified_file, 5)
        correctly_classified_seqs = read_sequences(correctly_classified_file, 5)

        for i, misclassified_seq in enumerate(misclassified_seqs):
            perform_msa(
                task, misclassified_seq, correctly_classified_seqs, output_folder, i
            )


def read_sequences(seq_file, num_seqs):
    """
    Read sequences from a FASTA file and return 'num_seqs' random sequences.

    Parameters:
    seq_file (str): Path to the input FASTA file.
    num_seqs (int): Number of sequences to randomly select.

    Returns:
    list: A list of randomly selected sequences.
    """
    with open(seq_file, "r") as file:
        lines = file.readlines()

    sequences = []
    current_seq = ""

    for line in lines:
        if line.startswith(">"):
            if current_seq:
                sequences.append(current_seq)
                current_seq = ""
        else:
            current_seq += line.strip()

    if current_seq:
        sequences.append(current_seq)

    return random.sample(sequences, min(num_seqs, len(sequences)))


# Example usage
tasks = [
    "transporters",
    "localization",
    "solubility",
    "ionchannels",
    "mp",
]

misclassified_folder = "./misclassified_sequences"
correctly_classified_folder = "./correctly_classified_sequences"
output_base_folder = "./task_alignments"

run_t_coffee_for_tasks(
    tasks, misclassified_folder, correctly_classified_folder, output_base_folder
)
