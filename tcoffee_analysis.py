import matplotlib.pyplot as plt
import numpy as np
import os
from Bio import AlignIO
import seaborn as sns 
from scipy.stats import entropy
import pandas as pd 


def calculate_conservation(msa):
    """Calculate conservation scores based on entropy for each position in the MSA."""
    # Transpose the MSA matrix to get columns (positions)
    transposed_msa = np.array([list(rec) for rec in msa], np.character).T

    # Calculate entropy for each position
    conservation_scores = []
    for column in transposed_msa:
        # Count the frequency of each amino acid in the column
        _, counts = np.unique(column, return_counts=True)
        # Normalize counts to probabilities
        probabilities = counts / counts.sum()
        # Calculate entropy and subtract from 1 to get conservation (1 - entropy)
        conservation_scores.append(1 - entropy(probabilities))

    return conservation_scores

# def plot_conservation(task, msa_conserved, msa_misclassified, output_folder, num_bins=10):
#     # Determine the maximum length of the two arrays
#     max_len = max(len(msa_conserved), len(msa_misclassified))

#     # Pad both arrays to be of max_len
#     msa_conserved = np.pad(msa_conserved, (0, max_len - len(msa_conserved)), 'constant', constant_values=np.nan)
#     msa_misclassified = np.pad(msa_misclassified, (0, max_len - len(msa_misclassified)), 'constant', constant_values=np.nan)

#     # Create a position array
#     positions = np.arange(max_len)

#     # Bin positions into 'num_bins' categories
#     bins = np.linspace(0, max_len, num_bins+1)
#     bin_labels = np.digitize(positions, bins) - 1

#     # Create a DataFrame
#     data = pd.DataFrame({
#         'Conservation Score': np.concatenate([msa_conserved, msa_misclassified]),
#         'Type': ['Correctly Classified']*len(msa_conserved) + ['Misclassified']*len(msa_misclassified),
#         'Bin': np.concatenate([bin_labels, bin_labels])
#     })

#     # Create a figure and plot the box plot
#     plt.figure(figsize=(12, 6))
#     sns.boxplot(x='Bin', y='Conservation Score', hue='Type', data=data, palette=['green', 'red'])

#     plt.xlabel('Bin of Alignment Position')
#     plt.ylabel('Conservation Score')
#     plt.title(f'Box Plots of Conservation Scores for {task}')
#     plt.legend(title='Classification')
#     plt.show()


def plot_conservation(task, msa_conserved, msa_misclassified, output_folder, num_bins=10):
    # Determine the maximum length of the two arrays
    max_len = max(len(msa_conserved), len(msa_misclassified))

    # Pad both arrays to be of max_len
    msa_conserved = np.pad(msa_conserved, (0, max_len - len(msa_conserved)), 'constant', constant_values=np.nan)
    msa_misclassified = np.pad(msa_misclassified, (0, max_len - len(msa_misclassified)), 'constant', constant_values=np.nan)

    # Create a position array
    positions = np.arange(max_len)

    # Bin positions into 'num_bins' categories
    bins = np.linspace(0, max_len, num_bins+1)
    bin_labels = np.digitize(positions, bins) - 1

    # Create a DataFrame
    data = pd.DataFrame({
        'Conservation Score': np.concatenate([msa_conserved, msa_misclassified]),
        'Type': ['Correctly Classified']*len(msa_conserved) + ['Misclassified']*len(msa_misclassified),
        'Bin': np.concatenate([bin_labels, bin_labels])
    })

    # Create a figure and plot the violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Bin', y='Conservation Score', hue='Type', data=data, split=True, inner='quartile', palette=['green', 'red'])

    plt.xlabel('Bin of Alignment Position')
    plt.ylabel('Conservation Score')
    # plt.title(f'Violin Plots of Conservation Scores for {task}')
    plt.legend(title='Classification')
    plt.savefig(os.path.join(output_folder, f"{task}_conservation_violin.png"), dpi=300)
    plt.close()
    


# def plot_conservation(task, msa_conserved, msa_misclassified, output_folder):
#     # Create a figure
#     plt.figure(figsize=(12, 6))
    
#     # Use seaborn to plot density
#     sns.kdeplot(msa_conserved, label='Correctly Classified', color='green', fill=True)
#     sns.kdeplot(msa_misclassified, label='Misclassified', color='red', fill=True)

#     plt.xlabel('Conservation Score')
#     plt.ylabel('Density')
#     plt.title(f'Density of Conservation Scores for {task}')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def plot_conservation(task, msa_conserved, msa_misclassified, output_folder):
#     # Determine the maximum length of the two arrays
#     max_len = max(len(msa_conserved), len(msa_misclassified))

#     # Pad both arrays to be of max_len
#     msa_conserved = np.pad(msa_conserved, (0, max_len - len(msa_conserved)), 'constant', constant_values=np.nan)
#     msa_misclassified = np.pad(msa_misclassified, (0, max_len - len(msa_misclassified)), 'constant', constant_values=np.nan)

#     # Create a 2D array where each row represents a dataset
#     data = np.vstack([msa_conserved, msa_misclassified])

#     # Create a figure and plot the heatmap
#     plt.figure(figsize=(16, 3))  # Adjust the size as needed
#     ax = sns.heatmap(data, cmap='viridis', linewidths=.5, annot=False, cbar_kws={'label': 'Conservation Score'})
#     ax.set_yticklabels(['Correctly Classified', 'Misclassified'])
#     ax.set_xlabel('Alignment Position')
#     ax.set_title(f'Heatmap of Conservation Scores for {task}')

#     plt.xticks(np.arange(0, max_len, step=max_len//10))  # Modify the step based on your data size
#     plt.show()


# def plot_conservation(task, msa_conserved, msa_misclassified, output_folder):
#     # Determine the maximum length of the two arrays
#     max_len = max(len(msa_conserved), len(msa_misclassified))

#     # Pad both arrays to be of max_len
#     msa_conserved = np.pad(msa_conserved, (0, max_len - len(msa_conserved)), 'constant', constant_values=np.nan)
#     msa_misclassified = np.pad(msa_misclassified, (0, max_len - len(msa_misclassified)), 'constant', constant_values=np.nan)

#     # Create a DataFrame for rolling mean
#     df = pd.DataFrame({
#         'Conserved': msa_conserved,
#         'Misclassified': msa_misclassified
#     })

#     # Apply rolling mean with a window size (e.g., 50 positions)
#     rolling_window_size = 50
#     smoothed_conserved = df['Conserved'].rolling(window=rolling_window_size, min_periods=1, center=True).mean()
#     smoothed_misclassified = df['Misclassified'].rolling(window=rolling_window_size, min_periods=1, center=True).mean()

#     # Create a figure
#     plt.figure(figsize=(12, 8))
#     plt.plot(smoothed_conserved, label='Correctly Classified', color='green')
#     plt.plot(smoothed_misclassified, label='Misclassified', color='red')

#     plt.xlabel('Alignment Position')
#     plt.ylabel('Smoothed Conservation Score')
#     plt.title(f'Smoothed Conservation Scores for {task}')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

    # plt.savefig(os.path.join(output_folder, f"{task}_conservation_line.png"))
    # plt.close()
    
# def plot_conservation(task, msa_conserved, msa_misclassified, output_folder):
#     # Determine the maximum length of the two arrays
#     max_len = max(len(msa_conserved), len(msa_misclassified))

#     # Pad both arrays to be of max_len
#     msa_conserved = np.pad(msa_conserved, (0, max_len - len(msa_conserved)), 'constant', constant_values=np.nan)
#     msa_misclassified = np.pad(msa_misclassified, (0, max_len - len(msa_misclassified)), 'constant', constant_values=np.nan)

#     # Positions array to match the padded arrays
#     positions = np.arange(max_len)

#     # Create a figure
#     plt.figure(figsize=(12, 8))

#     plt.scatter(positions, msa_conserved, c='green', alpha=0.6, edgecolors='w', s=50, label='Correctly Classified')
#     plt.scatter(positions, msa_misclassified, c='red', alpha=0.6, edgecolors='w', s=50, label='Misclassified')

#     plt.xlabel('Alignment Position')
#     plt.ylabel('Conservation Score')
#     # plt.title(f'Conservation Scores Scatter Plot for {task}')
#     plt.legend()
#     plt.grid(True)

#     plt.savefig(os.path.join(output_folder, f"{task}_conservation_scatter.png"))
#     plt.close()

def analyze_task(task, output_folder):
    msa_file_conserved = os.path.join(output_folder, task, f"{task}_msa_0.fasta")
    msa_file_misclassified = os.path.join(output_folder, task, f"{task}_msa_1.fasta")

    msa_conserved = AlignIO.read(msa_file_conserved, "fasta")
    msa_misclassified = AlignIO.read(msa_file_misclassified, "fasta")

    conserved_scores = calculate_conservation(msa_conserved)
    misclassified_scores = calculate_conservation(msa_misclassified)

    plot_conservation(task, conserved_scores, misclassified_scores, output_folder)

# Example usage
output_base_folder = "./task_alignments"
tasks = ["transporters", "localization", "solubility", "ionchannels", "mp"]

for task in tasks:
    analyze_task(task, output_base_folder)


# def calculate_conservation(msa):
#     """Calculate conservation scores based on entropy for each position in the MSA."""
#     # Transpose the MSA matrix to get columns (positions)
#     transposed_msa = np.array([list(rec) for rec in msa], np.character).T

#     # Calculate entropy for each position
#     conservation_scores = []
#     for column in transposed_msa:
#         # Count the frequency of each amino acid in the column
#         _, counts = np.unique(column, return_counts=True)
#         # Normalize counts to probabilities
#         probabilities = counts / counts.sum()
#         # Calculate entropy and subtract from 1 to get conservation (1 - entropy)
#         conservation_scores.append(1 - entropy(probabilities))

#     return conservation_scores

# def plot_conservation(task, msa_conserved, msa_misclassified, output_folder):
#     # Create a figure
#     plt.figure(figsize=(12, 6))
    
#     # Use seaborn to plot density
#     sns.kdeplot(msa_conserved, label='Correctly Classified', color='green', fill=True)
#     sns.kdeplot(msa_misclassified, label='Misclassified', color='red', fill=True)

#     plt.xlabel('Conservation Score')
#     plt.ylabel('Density')
#     # plt.title(f'Density of Conservation Scores for {task}')
#     plt.legend()
#     plt.grid(True)

#     plt.savefig(os.path.join(output_folder, f"{task}_density_plot.png"))
#     plt.close()

# def analyze_task(task, output_folder):
#     msa_file_conserved = os.path.join(output_folder, task, f"{task}_msa_0.fasta")
#     msa_file_misclassified = os.path.join(output_folder, task, f"{task}_msa_1.fasta")

#     msa_conserved = AlignIO.read(msa_file_conserved, "fasta")
#     msa_misclassified = AlignIO.read(msa_file_misclassified, "fasta")

#     conserved_scores = calculate_conservation(msa_conserved)
#     misclassified_scores = calculate_conservation(msa_misclassified)

#     plot_conservation(task, conserved_scores, misclassified_scores, output_folder)

# # Example usage
# output_base_folder = "./task_alignments"
# tasks = ["transporters", "localization", "solubility", "ionchannels", "mp"]

# for task in tasks:
#     analyze_task(task, output_base_folder)



# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from Bio import AlignIO

# def calculate_conservation(msa):
#     """Placeholder to simulate conservation scores."""
#     return np.random.rand(msa.get_alignment_length())

# def plot_conservation(task, msa_conserved, msa_misclassified, output_folder):
#     max_len = max(len(msa_conserved), len(msa_misclassified))
#     positions = np.arange(max_len)
    
#     # Pad the shorter array if necessary
#     msa_conserved = np.pad(msa_conserved, (0, max_len - len(msa_conserved)), 'constant')
#     msa_misclassified = np.pad(msa_misclassified, (0, max_len - len(msa_misclassified)), 'constant')

#     # Create a figure with subplots
#     fig, ax = plt.subplots(figsize=(12, 6))
#     width = 0.35  # Width of the bars
    
#     ax.bar(positions - width/2, msa_conserved, width, label='Correctly Classified', color='green')
#     ax.bar(positions + width/2, msa_misclassified, width, label='Misclassified', color='red')

#     ax.set_xlabel('Alignment Position')
#     ax.set_ylabel('Conservation Score')
#     # ax.set_title(f'Conservation Scores for {task}')
#     ax.legend()
#     ax.grid(True)

#     plt.savefig(os.path.join(output_folder, f"{task}_conservation_plot3.png"), dpi=300)
#     plt.close()

# def analyze_task(task, output_folder):
#     msa_file_conserved = os.path.join(output_folder, task, f"{task}_msa_0.fasta")
#     msa_file_misclassified = os.path.join(output_folder, task, f"{task}_msa_1.fasta")

#     msa_conserved = AlignIO.read(msa_file_conserved, "fasta")
#     msa_misclassified = AlignIO.read(msa_file_misclassified, "fasta")

#     conserved_scores = calculate_conservation(msa_conserved)
#     misclassified_scores = calculate_conservation(msa_misclassified)

#     plot_conservation(task, conserved_scores, misclassified_scores, output_folder)

# # Usage example
# output_base_folder = "./task_alignments"
# tasks = ["transporters", "localization", "solubility", "ionchannels", "mp"]

# for task in tasks:
#     analyze_task(task, output_base_folder)



# import os
# from Bio import AlignIO
# import matplotlib.pyplot as plt
# import numpy as np


# def calculate_conservation(msa):
#     """Calculate conservation scores for each position in the MSA."""
#     # Placeholder function; implement based on your chosen metric
#     return np.random.rand(msa.get_alignment_length())  # Example with random data


# def plot_conservation(task, msa_conserved, msa_misclassified, output_folder):
#     # Determine the maximum length of the two arrays
#     max_len = max(len(msa_conserved), len(msa_misclassified))

#     # Ensure both arrays are the same length by padding the shorter array
#     if len(msa_conserved) < max_len:
#         msa_conserved = np.pad(
#             msa_conserved, (0, max_len - len(msa_conserved)), "constant"
#         )
#     if len(msa_misclassified) < max_len:
#         msa_misclassified = np.pad(
#             msa_misclassified, (0, max_len - len(msa_misclassified)), "constant"
#         )

#     plt.figure(figsize=(12, 6))
#     positions = np.arange(max_len)
#     plt.plot(positions, msa_conserved, label="Correctly Classified", color="green")
#     plt.plot(positions, msa_misclassified, label="Misclassified", color="red")
#     # plt.title(f'Conservation Scores for {task}')
#     plt.xlabel("Alignment Position")
#     plt.ylabel("Conservation Score")
#     plt.legend()
#     plt.savefig(os.path.join(output_folder, f"{task}_conservation_plot.png"))
#     plt.close()


# def analyze_task(task, output_folder):
#     """Analyze MSA for a given task and create conservation plots."""
#     msa_file_conserved = os.path.join(
#         output_folder, task, f"{task}_msa_0.fasta"
#     )  # Adjust as needed
#     msa_file_misclassified = os.path.join(
#         output_folder, task, f"{task}_msa_1.fasta"
#     )  # Adjust as needed

#     # Parse the MSA output files
#     msa_conserved = AlignIO.read(msa_file_conserved, "fasta")
#     msa_misclassified = AlignIO.read(msa_file_misclassified, "fasta")

#     # Calculate conservation scores
#     conserved_scores = calculate_conservation(msa_conserved)
#     misclassified_scores = calculate_conservation(msa_misclassified)

#     # # print the conservation scores for task and misclassified and conserved
#     # print(f"Conservation scores for task {task}: {conserved_scores}")
#     # print(f"Conservation scores for misclassified {task}: {misclassified_scores}")

#     # Generate conservation plot
#     plot_conservation(task, conserved_scores, misclassified_scores, output_folder)


# # Example usage
# output_base_folder = "./task_alignments"
# tasks = ["transporters", "localization", "solubility", "ionchannels", "mp"]

# for task in tasks:
#     analyze_task(task, output_base_folder)

# import os
# from Bio import AlignIO
# import matplotlib.pyplot as plt
# import numpy as np


# def calculate_conservation(msa):
#     """Calculate conservation scores for each position in the MSA."""
#     # Placeholder function; implement based on your chosen metric
#     return np.random.rand(msa.get_alignment_length())  # Example with random data


# def plot_conservation(task, msa_conserved, msa_misclassified, output_folder):
#     # Determine the maximum length of the two arrays
#     max_len = max(len(msa_conserved), len(msa_misclassified))

#     # Ensure both arrays are the same length by padding the shorter array
#     if len(msa_conserved) < max_len:
#         msa_conserved = np.pad(
#             msa_conserved, (0, max_len - len(msa_conserved)), "constant"
#         )
#     if len(msa_misclassified) < max_len:
#         msa_misclassified = np.pad(
#             msa_misclassified, (0, max_len - len(msa_misclassified)), "constant"
#         )

#     plt.figure(figsize=(12, 6))
#     positions = np.arange(max_len)
#     plt.plot(positions, msa_conserved, label="Correctly Classified", color="#2ca02c")  # a clearer shade of green
#     plt.plot(positions, msa_misclassified, label="Misclassified", color="#d62728")  # a clearer shade of red
#     # plt.title(f'Conservation Scores for {task}')
#     plt.xlabel("Alignment Position")
#     plt.ylabel("Conservation Score")
#     plt.legend()
#     plt.grid(True)  # Adding grid
#     plt.tight_layout()  # Adjust layout to make room for labels and title
#     plt.savefig(os.path.join(output_folder, f"{task}_conservation_plot2.png"), dpi=300)  # Increase resolution
#     plt.close()


# def analyze_task(task, output_folder):
#     """Analyze MSA for a given task and create conservation plots."""
#     msa_file_conserved = os.path.join(
#         output_folder, task, f"{task}_msa_0.fasta"
#     )  # Adjust as needed
#     msa_file_misclassified = os.path.join(
#         output_folder, task, f"{task}_msa_1.fasta"
#     )  # Adjust as needed

#     # Parse the MSA output files
#     msa_conserved = AlignIO.read(msa_file_conserved, "fasta")
#     msa_misclassified = AlignIO.read(msa_file_misclassified, "fasta")

#     # Calculate conservation scores
#     conserved_scores = calculate_conservation(msa_conserved)
#     misclassified_scores = calculate_conservation(msa_misclassified)

#     # Generate conservation plot
#     plot_conservation(task, conserved_scores, misclassified_scores, output_folder)


# # Example usage
# output_base_folder = "./task_alignments"
# tasks = ["transporters", "localization", "solubility", "ionchannels", "mp"]

# for task in tasks:
#     analyze_task(task, output_base_folder)
