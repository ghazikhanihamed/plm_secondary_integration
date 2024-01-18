# Load the FASTA file and parse the sequences
from Bio import SeqIO

tasks = [
    "transporters",
    "localization",
    "solubility",
    "ionchannels",
    "mp",
]


# Define the path to the uploaded file
fasta_file_path = './task_alignments/ionchannels/ionchannels_msa_0.fasta'

# Function to parse the FASTA file and return sequences
def parse_fasta(fasta_file_path):
    with open(fasta_file_path, "r") as handle:
        fasta_records = list(SeqIO.parse(handle, "fasta"))
    return fasta_records

# Parse the FASTA file
sequences = parse_fasta(fasta_file_path)

# Count and differentiate between misclassified and correctly classified sequences
misclassified_seqs = [str(record.seq) for record in sequences if "misclassified" in record.id]
correctly_classified_seqs = [str(record.seq) for record in sequences if "correctly_classified" in record.id]

# Function to calculate sequence identity between two sequences
def sequence_identity(seq1, seq2):
    matches = sum(res1 == res2 for res1, res2 in zip(seq1, seq2))
    length = max(len(seq1), len(seq2))  # Use the total length of the alignment
    return matches / length if length > 0 else 0

# Calculate pairwise sequence identity between the misclassified sequence and each correctly classified sequence
misclassified_to_correctly_classified_identities = [
    sequence_identity(misclassified_seqs[0], seq) for seq in correctly_classified_seqs
]

# Calculate the average identity
avg_misclassified_to_correctly_classified_identity = sum(misclassified_to_correctly_classified_identities) / len(misclassified_to_correctly_classified_identities) if misclassified_to_correctly_classified_identities else 0

# Display the results
print("Average identity of misclassified sequence to correctly classified sequences:", avg_misclassified_to_correctly_classified_identity)
