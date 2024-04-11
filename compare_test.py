import pandas as pd

# Load the CSV file
file_path = './results/combined_results.csv'
df = pd.read_csv(file_path)

# Filter for test set evaluations (where fold is NaN) and for the two models of interest
test_eval_df = df[df['fold'].isna() & df['model'].isin(['ankh', 'p2s'])]

# Replace model names according to the requirement
test_eval_df['model'].replace({'ankh': 'Ankh', 'p2s': 'TooT-PLM-P2S'}, inplace=True)

# For SSP tasks, ensure the data_split reflects the dataset name correctly
test_eval_df.loc[test_eval_df['task'] == 'SSP', 'data_split'] = test_eval_df['ssp']

# Drop unnecessary columns for the comparison
comparison_df = test_eval_df.drop(columns=['fold', 'ssp'])

# Filter for a key metric, prioritizing 'eval_accuracy' when available
key_metric_df = comparison_df[((comparison_df['task'] != 'fluorescence') & (comparison_df['metric'] == 'eval_f1')) |
                              ((comparison_df['task'] == 'fluorescence') & (comparison_df['metric'] == 'eval_spearmanr'))]

# Pivot the data for easier comparison between models
pivot_df = key_metric_df.pivot_table(index=['task', 'data_split'], columns='model', values='value').reset_index()

# Sorting the tasks according to the specified order
task_order = [
    'fluorescence', 'solubility', 'localization', 'ionchannels', 'transporters',
    'mp', 'SSP_ssp3', 'SSP_ssp8'
]

# Sorting the tasks according to the specified order
pivot_df['task'] = pd.Categorical(pivot_df['task'], categories=task_order, ordered=True)
pivot_df.sort_values(['task', 'data_split'], inplace=True)
