from Bio import SeqIO

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
        fasta_file_path = f'./task_alignments/{task}/{task}_msa_{i}.fasta'
        
        # Parse the FASTA file
        sequences = parse_fasta(fasta_file_path)

        # Separate misclassified and correctly classified sequences
        misclassified_seqs = [str(record.seq) for record in sequences if "misclassified" in record.id]
        correctly_classified_seqs = [str(record.seq) for record in sequences if "correctly_classified" in record.id]

        # Calculate pairwise sequence identity for each misclassified to correctly classified sequence
        identities = [sequence_identity(misclassified_seqs[0], seq) for seq in correctly_classified_seqs]

        # Calculate the average identity for this file
        avg_identity = sum(identities) / len(identities) if identities else 0
        task_identities.append(avg_identity)

        print(f"Task: {task}, Alignment {i}, Average Identity: {avg_identity}")

    # Calculate and store the average identity for the task
    task_avg_identities[task] = sum(task_identities) / num_files

# Display the average identity for each task
for task, avg_identity in task_avg_identities.items():
    print(f"Task: {task}, Average Identity: {avg_identity}")

# Calculate and display the global average identity
global_avg_identity = sum(task_avg_identities.values()) / len(task_avg_identities)
print(f"Global Average Identity: {global_avg_identity}")
