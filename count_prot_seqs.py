from Bio import SeqIO

path_mis = "./misclassified_sequences/"
path_correct = "./correctly_classified_sequences/"

tasks = ["solubility", "localization", "ionchannels", "transporters", "mp"]


for task in tasks:
    misclassified_sequences = list(SeqIO.parse(f"{path_mis}/{task}_common_misclassified.fasta", "fasta"))
    correctly_classified_sequences = list(SeqIO.parse(f"{path_correct}/{task}_common_correctly_classified.fasta", "fasta"))
    
    print(f"Task: {task}")
    print(f"Misclassified sequences: {len(misclassified_sequences)}")
    print(f"Correctly classified sequences: {len(correctly_classified_sequences)}")
    print("\n")