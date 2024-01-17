import pandas as pd
from collections import Counter
from scipy.stats import chi2_contingency

# Load the CSV files for misclassified and correctly classified sequences
# Specifying the number of columns based on the data structure provided
column_names = [
    "query",
    "seed_ortholog",
    "evalue",
    "score",
    "eggNOG_OGs",
    "max_annot_lvl",
    "COG_category",
    "Description",
    "Preferred_name",
    "GOs",
    "EC",
    "KEGG_ko",
    "KEGG_Pathway",
    "KEGG_Module",
    "KEGG_Reaction",
    "KEGG_rclass",
    "BRITE",
    "KEGG_TC",
    "CAZy",
    "BiGG_Reaction",
    "PFAMs",
]

misclassified_df = pd.read_csv(
    "./eggNOG/mp_misclassified.emapper.annotations.tsv",
    sep="\t",
    names=column_names,
    skiprows=5,
    skipfooter=3,
)
correctly_classified_df = pd.read_csv(
    "./eggNOG/mp_correctly_classified.emapper.annotations.tsv",
    sep="\t",
    names=column_names,
    skiprows=5,
    skipfooter=3,
)


# Function to extract ENOG identifiers
def extract_enog_ids(df):
    return df["eggNOG_OGs"].str.split("@", expand=True)[0]


# Extract ENOGs from both datasets
misclassified_enogs = extract_enog_ids(misclassified_df)
correctly_classified_enogs = extract_enog_ids(correctly_classified_df)

# Count frequencies of ENOGs in both datasets
misclassified_enog_counts = misclassified_enogs.value_counts()
correctly_classified_enog_counts = correctly_classified_enogs.value_counts()

# Align the two series to have the same labels
aligned_misclassified, aligned_correctly = misclassified_enog_counts.align(
    correctly_classified_enog_counts, fill_value=0
)

# Now compare the frequencies
more_prevalent_in_misclassified = aligned_misclassified[
    aligned_misclassified > aligned_correctly
]

# Find exclusive ENOGs to misclassified sequences
exclusive_to_misclassified = misclassified_enog_counts.loc[
    ~misclassified_enog_counts.index.isin(correctly_classified_enog_counts.index)
]

# Display the results
print("Exclusive to misclassified:", exclusive_to_misclassified.index.tolist())
print("More prevalent in misclassified:", more_prevalent_in_misclassified)


# New function to parse the full hierarchical structure of ENOGs
def extract_full_enog_hierarchy(df):
    # Split the 'eggNOG_OGs' field and extract all parts
    return df["eggNOG_OGs"].str.split(",", expand=True)


# Extract the full hierarchy for both datasets
misclassified_hierarchy = extract_full_enog_hierarchy(misclassified_df)
correctly_classified_hierarchy = extract_full_enog_hierarchy(correctly_classified_df)

# Count the number of levels in the hierarchy for each dataset
misclassified_hierarchy_count = misclassified_hierarchy.count(axis=1)
correctly_classified_hierarchy_count = correctly_classified_hierarchy.count(axis=1)

# Display the results
print("Misclassified hierarchy levels:", misclassified_hierarchy_count.value_counts())
print(
    "Correctly classified hierarchy levels:",
    correctly_classified_hierarchy_count.value_counts(),
)


# New function to parse and count hierarchical levels
def count_hierarchy_levels(df):
    # Split and extract all parts of the 'eggNOG_OGs' field
    hierarchy = df["eggNOG_OGs"].str.split(",", expand=True)
    # Flatten the hierarchy and count occurrences of each level
    flattened_hierarchy = hierarchy.apply(
        lambda x: x.str.split("@").str[1]
    ).values.flatten()
    level_counts = Counter(flattened_hierarchy)
    return level_counts


# Count hierarchy levels in both datasets
misclassified_counts = count_hierarchy_levels(misclassified_df)
correctly_classified_counts = count_hierarchy_levels(correctly_classified_df)

# Display the results
print("Misclassified hierarchy levels:", misclassified_counts)
print("Correctly classified hierarchy levels:", correctly_classified_counts)


# Prepare data for chi-square test
# Assuming same levels are present in both datasets, otherwise, align them first
levels = list(
    set(misclassified_counts.keys()).union(set(correctly_classified_counts.keys()))
)
misclassified_freq = [misclassified_counts.get(level, 0) for level in levels]
correctly_classified_freq = [
    correctly_classified_counts.get(level, 0) for level in levels
]

# Chi-square test
chi2, p, dof, expected = chi2_contingency(
    [misclassified_freq, correctly_classified_freq]
)

# Display the results
print("Chi-square test result:")
print(f"Chi-square Statistic: {chi2}, P-value: {p}")

# Interpret the results
if p < 0.05:
    print(
        "Significant difference in the distribution of hierarchical levels between misclassified and correctly classified sequences."
    )
else:
    print("No significant difference in the distribution of hierarchical levels.")
