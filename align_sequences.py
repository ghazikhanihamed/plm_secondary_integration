import os
import subprocess
import random


def prepend_id_prefix(file_path, prefix):
    with open(file_path, "r") as file:
        lines = file.readlines()
    with open(file_path, "w") as file:
        for line in lines:
            if line.startswith(">"):
                file.write(f">{prefix}{line[1:]}")
            else:
                file.write(line)


def perform_msa(task, seq_file1, seq_files_group, output_path):
    # Combine the single sequence with the group of sequences into one file
    combined_file = f"{task}_combined_temp.fasta"
    with open(combined_file, "w") as outfile:
        with open(seq_file1) as infile:
            outfile.write(infile.read())
            outfile.write("\n")
        for seq_file in seq_files_group:
            with open(seq_file) as infile:
                outfile.write(infile.read())
                outfile.write("\n")

    # Run T-Coffee for MSA
    command = f"t_coffee -in {combined_file} -outfile {output_path} -output fasta"
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

        # Prepend unique prefixes to sequence IDs in each file
        prepend_id_prefix(misclassified_file, "mis_")
        prepend_id_prefix(correctly_classified_file, "cor_")

        # Read sequences from files and randomly select five from each
        misclassified_seqs = read_sequences(misclassified_file, 5)
        correctly_classified_seqs = read_sequences(correctly_classified_file, 5)

        for i, misclassified_seq in enumerate(misclassified_seqs):
            output_path = os.path.join(output_folder, f"alignment_{i+1}.fasta")
            perform_msa(task, misclassified_seq, correctly_classified_seqs, output_path)


def read_sequences(seq_file, num_seqs):
    # Read sequences from a file and return 'num_seqs' random sequences
    with open(seq_file, "r") as file:
        sequences = [
            line.strip()
            for line in file.readlines()
            if line.strip() and not line.startswith(">")
        ]
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
