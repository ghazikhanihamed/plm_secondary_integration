import pandas as pd
import os


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


def convert_all_csv_to_fasta(folder_path, seq_col):
    """
    Convert all CSV files in a folder to FASTA format.

    Parameters:
    folder_path (str): Path to the folder containing CSV files.
    seq_col (str): Column name in CSV files that contains sequences.
    """
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            csv_file = os.path.join(folder_path, file)
            fasta_file = os.path.join(folder_path, file.replace(".csv", ".fasta"))
            csv_to_fasta(csv_file, fasta_file, seq_col)


# Run the function
folder_path = "./correctly_classified_sequences" 
seq_col = "sequence" 

convert_all_csv_to_fasta(folder_path, seq_col)
