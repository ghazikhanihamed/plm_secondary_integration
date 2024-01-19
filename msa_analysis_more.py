import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
from collections import Counter
import numpy as np
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


# Function to calculate a summary statistic from amino acid composition
def summarize_aa_composition(aa_composition):
    summary_stats = []
    for pos in aa_composition:
        # Example: calculate the mean frequency of a specific amino acid at each position
        # Modify this as per your specific analysis needs
        if "A" in aa_composition[pos]:  # Assuming 'A' is the amino acid of interest
            summary_stats.append(aa_composition[pos]["A"])
        else:
            summary_stats.append(0)
    return summary_stats


# Function to calculate sequence conservation
def calculate_conservation(sequences):
    # Convert sequences to a matrix where each row is a sequence and each column is a position
    seq_matrix = np.array([list(seq) for seq in sequences])
    num_seqs, seq_length = seq_matrix.shape

    conservation_scores = np.zeros(seq_length)

    # Iterate over each position
    for pos in range(seq_length):
        # Count the frequency of each amino acid at this position
        aa_freq = Counter(seq_matrix[:, pos])

        # Ignore gaps in the conservation score calculation
        if "-" in aa_freq:
            del aa_freq["-"]

        # Calculate the frequency of the most common amino acid
        if aa_freq:
            most_common_freq = aa_freq.most_common(1)[0][1] / num_seqs
            conservation_scores[pos] = most_common_freq

    return conservation_scores


# Function to analyze gap distribution
def analyze_gaps(sequences):
    # Convert sequences to a matrix where each row is a sequence and each column is a position
    seq_matrix = np.array([list(seq) for seq in sequences])
    num_seqs, seq_length = seq_matrix.shape

    gap_distribution = np.zeros(seq_length)

    # Iterate over each position
    for pos in range(seq_length):
        # Count the number of gaps at this position
        gap_count = np.sum(seq_matrix[:, pos] == "-")
        gap_distribution[pos] = gap_count / num_seqs

    return gap_distribution


# Function to analyze amino acid composition
def amino_acid_composition(sequences):
    # Convert sequences to a matrix where each row is a sequence and each column is a position
    seq_matrix = np.array([list(seq) for seq in sequences])
    _, seq_length = seq_matrix.shape

    # Initialize a dictionary to store amino acid frequencies for each position
    aa_composition = {pos: Counter() for pos in range(seq_length)}

    # Iterate over each position
    for pos in range(seq_length):
        # Count the amino acids at this position, ignoring gaps
        aa_composition[pos].update(filter(lambda aa: aa != "-", seq_matrix[:, pos]))

    # Convert counts to frequencies
    for pos in aa_composition:
        total_count = sum(aa_composition[pos].values())
        if total_count > 0:
            for aa in aa_composition[pos]:
                aa_composition[pos][aa] /= total_count

    return aa_composition


# Function to perform statistical comparison
def perform_statistical_tests(metric_misclassified, metric_correctly_classified):
    """
    Perform statistical comparison between misclassified and correctly classified sequences.

    Args:
        metric_misclassified (list): A list of values (e.g., conservation scores) for misclassified sequences.
        metric_correctly_classified (list): A list of values for correctly classified sequences.

    Returns:
        A dictionary containing the p-values of the performed tests.
    """
    results = {}

    # T-test (for normally distributed data)
    t_stat, t_p_value = ttest_ind(
        metric_misclassified, metric_correctly_classified, equal_var=False
    )
    results["t-test"] = t_p_value

    # Mann-Whitney U test (for non-normally distributed data)
    mw_stat, mw_p_value = mannwhitneyu(
        metric_misclassified, metric_correctly_classified, alternative="two-sided"
    )
    results["mann-whitney"] = mw_p_value

    return results


# Function to separate misclassified and correctly classified sequences
def separate_sequences(sequences):
    misclassified_seqs = [
        str(record.seq) for record in sequences if "misclassified" in record.id
    ]
    correctly_classified_seqs = [
        str(record.seq) for record in sequences if "correctly_classified" in record.id
    ]
    return misclassified_seqs, correctly_classified_seqs


# Function to analyze alignments for unique patterns
def analyze_alignment(sequences):
    misclassified_seqs, correctly_classified_seqs = separate_sequences(sequences)

    # Perform analysis on each sequence type
    conservation_scores_misclassified = calculate_conservation(misclassified_seqs)
    conservation_scores_correctly_classified = calculate_conservation(
        correctly_classified_seqs
    )

    gap_distribution_misclassified = analyze_gaps(misclassified_seqs)
    gap_distribution_correctly_classified = analyze_gaps(correctly_classified_seqs)

    aa_composition_misclassified = amino_acid_composition(misclassified_seqs)
    aa_composition_correctly_classified = amino_acid_composition(
        correctly_classified_seqs
    )

    # Handle amino acid composition
    aa_composition_misclassified = amino_acid_composition(misclassified_seqs)
    aa_composition_correctly_classified = amino_acid_composition(
        correctly_classified_seqs
    )

    # Summarize the amino acid compositions
    aa_summary_misclassified = summarize_aa_composition(aa_composition_misclassified)
    aa_summary_correctly_classified = summarize_aa_composition(
        aa_composition_correctly_classified
    )

    # Perform statistical tests
    conservation_test_results = perform_statistical_tests(
        conservation_scores_misclassified, conservation_scores_correctly_classified
    )
    gap_test_results = perform_statistical_tests(
        gap_distribution_misclassified, gap_distribution_correctly_classified
    )
    aa_test_results = perform_statistical_tests(
        aa_summary_misclassified, aa_summary_correctly_classified
    )

    return {
        "conservation": conservation_test_results,
        "gaps": gap_test_results,
        "amino_acids": aa_test_results,
    }


# Initialize dictionaries to store the aggregated results
overall_results = {
    "conservation": {"t-test": [], "mann-whitney": []},
    "gaps": {"t-test": [], "mann-whitney": []},
    "amino_acids": {"t-test": [], "mann-whitney": []},
}

# Main code to process each task and its alignments
for task, num_files in tasks.items():
    print(f"Task: {task}")

    for i in range(num_files):
        # Construct the file path and parse the FASTA file
        fasta_file_path = f"./task_alignments/{task}/{task}_msa_{i}.fasta"
        sequences = parse_fasta(fasta_file_path)

        # Analyze the alignment
        analysis_results = analyze_alignment(sequences)
        print(f"Alignment {i}: {analysis_results}")

        # Aggregate the results
        for metric in analysis_results:
            for test in analysis_results[metric]:
                overall_results[metric][test].append(analysis_results[metric][test])

# Compute and display the overall summary
print("\nOverall Summary Across All Tasks:")
for metric, tests in overall_results.items():
    print(f"\nMetric: {metric}")
    for test, results in tests.items():
        # Calculate the average p-value for each test
        average_p_value = sum(results) / len(results) if results else None
        print(f"  {test} average p-value: {average_p_value}")
