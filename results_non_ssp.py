import pandas as pd

# Load the CSV file
file_path = './results/cross_validation_results.csv'
data = pd.read_csv(file_path)

# Display the first few rows to understand the structure
data.head()

# Filter the data for non-SSP tasks and calculate mean and standard deviation for each metric
non_ssp_tasks = data[~data['task'].str.contains('ssp')]
non_ssp_summary = non_ssp_tasks.groupby(['model', 'task', 'metric', 'data_split']).agg(
    mean_value=('value', 'mean'),
    std_value=('value', 'std')
).reset_index()

import matplotlib.pyplot as plt
import seaborn as sns

# Create tables and figures for non-SSP tasks
table_non_ssp_cv = non_ssp_summary[non_ssp_summary['data_split'] == 'cross-validation']
table_non_ssp_test = non_ssp_summary[non_ssp_summary['data_split'] == 'test']

# Display the summarized table for cross-validation results
table_non_ssp_cv_pivot = table_non_ssp_cv.pivot_table(
    index=['task', 'metric'],
    columns='model',
    values=['mean_value', 'std_value']
).round(4)

# Display the summarized table for test set results
table_non_ssp_test_pivot = table_non_ssp_test.pivot_table(
    index=['task', 'metric'],
    columns='model',
    values=['mean_value', 'std_value']
).round(4)


# Plotting the results for cross-validation
plt.figure(figsize=(14, 8))
sns.barplot(
    data=non_ssp_tasks[non_ssp_tasks['data_split'] == 'cross-validation'],
    x='task',
    y='value',
    hue='model',
    ci='sd'
)
plt.title('Cross-Validation Performance for Non-SSP Tasks')
plt.xlabel('Task')
plt.ylabel('Metric Value')
plt.legend(title='Model')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig('/mnt/data/non_ssp_tasks_cv_performance.png')
plt.show()

# Plotting the results for test set
plt.figure(figsize=(14, 8))
sns.barplot(
    data=non_ssp_tasks[non_ssp_tasks['data_split'] == 'test'],
    x='task',
    y='value',
    hue='model',
    ci='sd'
)
plt.title('Test Set Performance for Non-SSP Tasks')
plt.xlabel('Task')
plt.ylabel('Metric Value')
plt.legend(title='Model')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig('/mnt/data/non_ssp_tasks_test_performance.png')
plt.show()
