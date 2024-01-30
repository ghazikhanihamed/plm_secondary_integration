# Re-importing necessary libraries after code execution state reset
import pandas as pd

# Load the provided results data to inspect and analyze for areas where TooT-PLM-P2S outperforms Ankh
file_path = './results/combined_results.csv'
data = pd.read_csv(file_path)

# Replace model names for consistency
data['model'].replace({'p2s': 'TooT-PLM-P2S', 'ankh': 'Ankh'}, inplace=True)

# Filter for cross-validation results only
cv_data = data[data['data_split'] == 'cross-validation'].copy()

# Determine where TooT-PLM-P2S outperforms Ankh by calculating the difference in performance
# Group by task and metric, then pivot the table to compare the two models
performance_comparison = cv_data.pivot_table(index=['task', 'metric'], columns='model', values='value', aggfunc='mean')

# Calculate the performance difference (TooT-PLM-P2S - Ankh)
performance_comparison['Performance Difference'] = performance_comparison['TooT-PLM-P2S'] - performance_comparison['Ankh']

# Filter to show only the tasks and metrics where TooT-PLM-P2S has better performance
better_performance = performance_comparison[performance_comparison['Performance Difference'] > 0]

print(better_performance)
