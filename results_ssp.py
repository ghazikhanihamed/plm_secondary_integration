import pandas as pd

# Load the CSV file
ssp_file_path = './results/cross_validation_results_ssp.csv'
ssp_data = pd.read_csv(ssp_file_path)

# Display the first few rows to understand the structure
ssp_data.head()

# Filter the data for ssp3 and ssp8 tasks and calculate mean and standard deviation for each metric
ssp_summary = ssp_data.groupby(['model', 'ssp', 'metric', 'data_split']).agg(
    mean_value=('value', 'mean'),
    std_value=('value', 'std')
).reset_index()

import matplotlib.pyplot as plt
import seaborn as sns

# Create tables and figures for SSP tasks
table_ssp_cv = ssp_summary[ssp_summary['data_split'] == 'cross-validation']
table_ssp_test = ssp_summary[ssp_summary['data_split'] != 'cross-validation']

# Display the summarized table for cross-validation results
table_ssp_cv_pivot = table_ssp_cv.pivot_table(
    index=['ssp', 'metric'],
    columns='model',
    values=['mean_value', 'std_value']
).round(4)

# Display the summarized table for test set results
table_ssp_test_pivot = table_ssp_test.pivot_table(
    index=['ssp', 'data_split', 'metric'],
    columns='model',
    values=['mean_value', 'std_value']
).round(4)

# Create figures
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

# Plot for SSP3 Accuracy
sns.barplot(x='data_split', y='mean_value', hue='model', data=table_ssp_test[(table_ssp_test['ssp'] == 'ssp3') & (table_ssp_test['metric'] == 'eval_accuracy')], ax=axes[0, 0])
axes[0, 0].set_title('SSP3 Accuracy')
axes[0, 0].set_ylabel('Accuracy')

# Plot for SSP3 F1 Score
sns.barplot(x='data_split', y='mean_value', hue='model', data=table_ssp_test[(table_ssp_test['ssp'] == 'ssp3') & (table_ssp_test['metric'] == 'eval_f1')], ax=axes[0, 1])
axes[0, 1].set_title('SSP3 F1 Score')
axes[0, 1].set_ylabel('F1 Score')

# Plot for SSP8 Accuracy
sns.barplot(x='data_split', y='mean_value', hue='model', data=table_ssp_test[(table_ssp_test['ssp'] == 'ssp8') & (table_ssp_test['metric'] == 'eval_accuracy')], ax=axes[1, 0])
axes[1, 0].set_title('SSP8 Accuracy')
axes[1, 0].set_ylabel('Accuracy')

# Plot for SSP8 F1 Score
sns.barplot(x='data_split', y='mean_value', hue='model', data=table_ssp_test[(table_ssp_test['ssp'] == 'ssp8') & (table_ssp_test['metric'] == 'eval_f1')], ax=axes[1, 1])
axes[1, 1].set_title('SSP8 F1 Score')
axes[1, 1].set_ylabel('F1 Score')

plt.tight_layout()
plt.savefig('./plots/ssp_results.png', dpi=300, bbox_inches='tight')
plt.show()
