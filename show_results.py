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

# Setting the appropriate metric for each task
# For 'fluorescence', use 'eval_spearmanr'
cv_data.loc[cv_data['task'] == 'fluorescence', 'metric'] = 'eval_spearmanr'

# For 'ssp', distinguish between 'ssp3' and 'ssp8' and use 'eval_accuracy'
cv_data.loc[(cv_data['task'] == 'ssp') & (cv_data['ssp'] == 'ssp3'), 'metric'] = 'eval_accuracy'
cv_data.loc[(cv_data['task'] == 'ssp') & (cv_data['ssp'] == 'ssp8'), 'metric'] = 'eval_accuracy'

# For all other tasks, use 'eval_mcc'
for task in cv_data['task'].unique():
    if task not in ['fluorescence', 'ssp']:
        cv_data.loc[cv_data['task'] == task, 'metric'] = 'eval_mcc'

# # Adjust the task column for SSP tasks to include SSP type
# cv_data.loc[cv_data['task'] == 'ssp', 'task'] = cv_data['task'] + '_' + cv_data['ssp']

# Now filter the data based on the metric conditions specified
filtered_data = cv_data[
    ((cv_data['task'] == 'fluorescence') & (cv_data['metric'] == 'eval_spearmanr')) |
    (cv_data['task'].str.contains('ssp') & (cv_data['metric'] == 'eval_accuracy')) |
    (~cv_data['task'].str.contains('ssp|fluorescence') & (cv_data['metric'] == 'eval_mcc'))
]

# Setting up the figure for plotting
plt.figure(figsize=(14, 8))

# Creating a barplot for comparison of models across different tasks including SSP types
sns.barplot(x='task', y='value', hue='model', data=filtered_data, palette='viridis')

# Adding labels and title to the plot
plt.xlabel('Task', fontsize=14)
plt.ylabel('Metric Value', fontsize=14)
plt.title('Comparative Performance Overview Across Tasks Including SSP Types', fontsize=16)
plt.xticks(rotation=45)
plt.legend(title='Model')

# Display the plot
plt.tight_layout()
plt.show()

# Preparing data for the table: Aggregating metric values by model and task including SSP types
summary_table = filtered_data.groupby(['model', 'task', 'metric'])['value'].mean().unstack().unstack()

# Generating the LaTeX table with the summary data
latex_table = summary_table.to_latex(float_format="%.3f", header=True, index=True, 
                                     caption="Comparative Performance Overview Across Tasks Including SSP Types",
                                     label="tab:performance_overview_ssp", position="ht",
                                     column_format='lccc', bold_rows=True)

print(latex_table)
