import pandas as pd


def csv_to_fasta(csv_file, fasta_file, seq_col):
    """
    Convert sequences from a CSV file to a FASTA format file.
    The sequence ID in the FASTA file will be the row index from the CSV file.

    Parameters:
    csv_file (str): Path to the input CSV file.
    fasta_file (str): Path to the output FASTA file.
    seq_col (str): Column name in CSV file that contains sequences.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Open the FASTA file for writing
    with open(fasta_file, "w") as f:
        # Iterate over the rows in DataFrame
        for index, row in df.iterrows():
            # Write the FASTA format (">" followed by row index and then sequence on next line)
            f.write(f">{index}\n{row[seq_col]}\n")


# Example usage
csv_file = "path/to/your/csvfile.csv"  # Replace with your CSV file path
fasta_file = "path/to/output.fasta"  # Replace with your desired output path
seq_col = "sequence"  # Replace with your sequence column name

csv_to_fasta(csv_file, fasta_file, seq_col)
