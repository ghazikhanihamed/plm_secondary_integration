import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the CSV file to inspect the data and understand its structure
file_path = './results/combined_results.csv'
data = pd.read_csv(file_path)

# Adjusting the code based on the new requirements

# Preprocessing the data for visualization and analysis
data['model'].replace({'p2s': 'TooT-PLM-P2S', 'ankh': 'Ankh'}, inplace=True)

# Filtering data for cross-validation results only
cv_data = data[data['data_split'] == 'cross-validation'].copy()

# Adjusting the code to create three different plots: one for MCC, one for Fluorescence, and separate ones for SSP3 and SSP8

# Define a function for plotting
def plot_performance(data, title, ylabel, is_ssp=False):
    plt.figure(figsize=(12, 6))
    x_axis = 'Task_SSP' if is_ssp else 'task'
    
    barplot = sns.barplot(x=x_axis, y='value', hue='model', data=data, palette='viridis', ci='sd')

    # Annotating each bar with its value
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.3f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center',
                         xytext=(0, 10), textcoords='offset points')

    plt.xlabel('Task', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45)
    plt.legend(title='Model', loc='upper right')
    plt.tight_layout()
    plt.show()


# Plot for tasks with MCC metric
mcc_data = cv_data[(~cv_data['task'].str.contains('fluorescence|ssp')) & (cv_data['metric'] == 'eval_mcc')]
plot_performance(mcc_data, 'Performance Overview Across Tasks with MCC', 'MCC')

# Plot for Fluorescence with Spearman's correlation
fluorescence_data = cv_data[(cv_data['task'] == 'fluorescence') & (cv_data['metric'] == 'eval_spearmanr')]
plot_performance(fluorescence_data, 'Fluorescence Task Performance (Spearman\'s Correlation)', 'Spearman\'s Correlation')

# Separate plots for SSP3 and SSP8 with accuracy
ssp3_data = cv_data[(cv_data['task'] == 'SSP') & (cv_data['metric'] == 'eval_accuracy') & (cv_data['ssp'] == 'ssp3')]
ssp8_data = cv_data[(cv_data['task'] == 'SSP') & (cv_data['metric'] == 'eval_accuracy') & (cv_data['ssp'] == 'ssp8')]

# Combine SSP3 and SSP8 data
ssp_data = pd.concat([ssp3_data, ssp8_data])

# Ensure 'SSP Type' column is correctly added to distinguish between SSP3 and SSP8
ssp_data['SSP Type'] = ssp_data['ssp']

# Ensure 'Task_SSP' column combines task name with SSP type for clear distinction
ssp_data['Task_SSP'] = ssp_data['task'] + ' (' + ssp_data['SSP Type'] + ')'

# Now plot the combined SSP3 and SSP8 data using the adjusted plot_performance function
plot_performance(ssp_data, 'Comparative Performance on SSP3 and SSP8 Tasks', 'Accuracy', is_ssp=True)


# Ensure the 'value' column is numeric for aggregation
cv_data['value'] = pd.to_numeric(cv_data['value'], errors='coerce')

# Selecting the appropriate metric for each task
cv_data['selected_metric'] = cv_data.apply(
    lambda x: 'eval_spearmanr' if x['task'] == 'fluorescence' else (
        'eval_accuracy' if 'ssp' in x['task'] else 'eval_mcc'), axis=1)

# Update task names for SSP3 and SSP8 and format metrics correctly
cv_data['task'] = cv_data.apply(lambda x: f"{x['task']}_ssp3" if x['ssp'] == 'ssp3' else (f"{x['task']}_ssp8" if x['ssp'] == 'ssp8' else x['task']), axis=1)
cv_data['selected_metric'] = cv_data['selected_metric'].str.replace('eval_', '').str.upper()
cv_data['selected_metric'] = cv_data['selected_metric'].replace({'MCC': 'MCC', 'ACCURACY': 'Accuracy', 'SPEARMANR': 'SpearmanR'})

# Recalculate mean and standard deviation for each task, model, and selected metric
grouped_data = cv_data.groupby(['task', 'model', 'selected_metric'])['value'].agg(['mean', 'std']).reset_index()

# Create a performance column displaying mean ± std
grouped_data['Performance'] = grouped_data.apply(lambda x: f"{x['mean']:.3f} ± {x['std']:.3f}", axis=1)

# Pivot the table for a side-by-side comparison
pivot_table = grouped_data.pivot_table(index=['task', 'selected_metric'], columns='model', values='Performance', aggfunc='first').reset_index()

# Convert the pivot table to LaTeX format
latex_table = pivot_table.to_latex(index=False, escape=False, column_format='llcc', float_format="%.3f")

# Display the LaTeX table
print(latex_table)

