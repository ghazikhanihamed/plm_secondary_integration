import pandas as pd

# Read in the results from the first cross-validation
cv_results = pd.read_csv('./results/cross_validation_results.csv')
cv_results_ssp = pd.read_csv('./results/cross_validation_results_ssp.csv')

cv_results['ssp'] = 'NA'  # Assign a default value for 'ssp' in the first dataframe

# Now we can concatenate the two dataframes
combined_results = pd.concat([cv_results, cv_results_ssp], ignore_index=True)

# save the combined results
combined_results.to_csv('./results/combined_results.csv', index=False)

a=1
