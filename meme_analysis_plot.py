import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Function to extract motif information from the text file
def extract_motif_info_from_txt(file_path):
    motifs = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        motif_id = None
        for line in lines:
            if line.startswith('MOTIF'):
                motif_id = line.split()[1]
                motifs[motif_id] = {'occurrences': 0}
            elif motif_id and line.startswith(' '):
                motifs[motif_id]['occurrences'] += 1
    return motifs

# List of tasks
tasks = ["ionchannels", "localization", "mp", "solubility", "transporters"]

# Base path where the meme files are located
base_path = './meme_files/'

# Initialize an empty DataFrame for the box plot data
box_plot_data = pd.DataFrame(columns=['Task', 'Classification', 'Occurrences'])

# Loop through each task
for task in tasks:
    # Construct file paths
    misclassified_txt_path = f'{base_path}/{task}_common_misclassified_meme.txt'
    correctly_classified_txt_path = f'{base_path}/{task}_common_correctly_classified_meme.txt'
    
    # Extract motifs
    misclassified_motifs = extract_motif_info_from_txt(misclassified_txt_path)
    correctly_classified_motifs = extract_motif_info_from_txt(correctly_classified_txt_path)
    
    # Append the motif occurrences to the DataFrame
    for motif, info in misclassified_motifs.items():
        box_plot_data = box_plot_data._append({'Task': task, 'Classification': 'Misclassified', 'Occurrences': info['occurrences']}, ignore_index=True)
    for motif, info in correctly_classified_motifs.items():
        box_plot_data = box_plot_data._append({'Task': task, 'Classification': 'Correctly Classified', 'Occurrences': info['occurrences']}, ignore_index=True)
        
# print the box plot data
print(box_plot_data)

# Plotting the box plot
plt.figure(figsize=(14, 8))
sns.boxplot(x='Task', y='Occurrences', hue='Classification', data=box_plot_data, palette="Set2")
# plt.title('Distribution of Motif Occurrences Across Tasks')
plt.xlabel('Tasks')
plt.ylabel('Occurrences of Motifs')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(f'./{base_path}/motif_occurrences_boxplot.png', bbox_inches='tight', dpi=300)

# Show the plot
# plt.show()

# # Initialize an empty DataFrame for the heatmap data
# heatmap_data = pd.DataFrame()

# # Loop through each task
# for task in tasks:
#     # Construct file paths
#     misclassified_txt_path = f'{base_path}/{task}_common_misclassified_meme.txt'
#     correctly_classified_txt_path = f'{base_path}/{task}_common_correctly_classified_meme.txt'
    
#     # Extract motifs
#     misclassified_motifs = extract_motif_info_from_txt(misclassified_txt_path)
#     correctly_classified_motifs = extract_motif_info_from_txt(correctly_classified_txt_path)
    
#     # Merge the motif data
#     for motif, info in misclassified_motifs.items():
#         heatmap_data.loc[motif, f'{task}_misclassified'] = info['occurrences']
#     for motif, info in correctly_classified_motifs.items():
#         heatmap_data.loc[motif, f'{task}_correctly_classified'] = info['occurrences']

# # Replace NaN values with 0
# heatmap_data.fillna(0, inplace=True)

# # Log transform the data to better visualize the range of occurrences
# heatmap_data_log = np.log1p(heatmap_data)

# # Plotting the heatmap
# plt.figure(figsize=(14, 10))
# sns.heatmap(heatmap_data_log, annot=True, fmt=".1f", cmap='viridis', linewidths=.5)
# plt.title('Log-Scaled Occurrences of Motifs in Misclassified vs. Correctly Classified Sequences')
# plt.ylabel('Motifs')
# plt.xlabel('Tasks and Classification')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()

# # Show the plot
# plt.show()

# # Loop through each task
# for task in tasks:
#     # Construct file paths
#     misclassified_txt_path = f'{base_path}/{task}_common_misclassified_meme.txt'
#     correctly_classified_txt_path = f'{base_path}/{task}_common_correctly_classified_meme.txt'
    
#     # Extract motifs
#     misclassified_motifs_info = extract_motif_info_from_txt(misclassified_txt_path)
#     correctly_classified_motifs_info = extract_motif_info_from_txt(correctly_classified_txt_path)
    
#     # Prepare data for plotting
#     motifs = list(set(misclassified_motifs_info.keys()) | set(correctly_classified_motifs_info.keys()))
#     misclassified_counts = [misclassified_motifs_info.get(motif, {'occurrences': 0})['occurrences'] for motif in motifs]
#     correctly_classified_counts = [correctly_classified_motifs_info.get(motif, {'occurrences': 0})['occurrences'] for motif in motifs]
    
#     # Plotting
#     x = np.arange(len(motifs))  # the label locations
#     width = 0.35  # the width of the bars

#     fig, ax = plt.subplots(figsize=(12, 6))
#     rects1 = ax.bar(x - width/2, misclassified_counts, width, label='Misclassified', color='tomato')
#     rects2 = ax.bar(x + width/2, correctly_classified_counts, width, label='Correctly Classified', color='lightgreen')

#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     ax.set_ylabel('Occurrences')
#     ax.set_title(f'Comparison of Motif Occurrences in {task.capitalize()} Misclassified vs. Correctly Classified Sequences')
#     ax.set_xticks(x)
#     ax.set_xticklabels(motifs, rotation=45)
#     ax.legend()

#     ax.bar_label(rects1, padding=3)
#     ax.bar_label(rects2, padding=3)

#     plt.tight_layout()

#     # Show the plot
#     plt.show()
