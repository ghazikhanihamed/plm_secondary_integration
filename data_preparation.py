from datasets import load_dataset
import pandas as pd
import os


def load_and_combine_hf_datasets(
    dataset_name,
    train_input_column,
    train_label_column,
    test_input_column,
    test_label_column,
    output_dir=".",
):
    # Load dataset from Hugging Face
    dataset = load_dataset(dataset_name)

    # Extract training set
    training_sequences = dataset["train"][train_input_column]
    training_labels = dataset["train"][train_label_column]

    # Initialize validation sequences and labels
    validation_sequences = []
    validation_labels = []

    # Check if there is an existing validation set and combine it with the training set
    if "validation" in dataset:
        validation_sequences = dataset["validation"][train_input_column]
        validation_labels = dataset["validation"][train_label_column]
        training_sequences += validation_sequences
        training_labels += validation_labels

    # Extract test set
    test_sequences = dataset["test"][test_input_column]
    test_labels = dataset["test"][test_label_column]

    # Save the combined training set to a CSV file
    train_df = pd.DataFrame(
        {train_input_column: training_sequences, train_label_column: training_labels}
    )
    train_df.to_csv(f"{output_dir}/train.csv", index=False)

    # Save the test set to a CSV file
    test_df = pd.DataFrame(
        {test_input_column: test_sequences, test_label_column: test_labels}
    )
    test_df.to_csv(f"{output_dir}/test.csv", index=False)


def load_and_save_secondary_structure_datasets(output_dir="."):
    # Load training dataset
    training_dataset = load_dataset(
        "proteinea/SSP", data_files={"train": ["training_hhblits.csv"]}
    )

    input_column_name = "input"
    dssp3_column_name = "dssp3"
    dssp8_column_name = "dssp8"
    disorder_column_name = "disorder"

    # Save training set to CSV, including both dssp3 and dssp8
    train_df = pd.DataFrame(
        {
            input_column_name: training_dataset["train"][input_column_name],
            dssp3_column_name: training_dataset["train"][dssp3_column_name],
            dssp8_column_name: training_dataset["train"][dssp8_column_name],
            disorder_column_name: training_dataset["train"][disorder_column_name],
        }
    )
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)

    # Load and save test datasets, including both dssp3 and dssp8
    test_datasets = {
        "casp12": "CASP12.csv",
        "casp13": "CASP13.csv",
        "casp14": "CASP14.csv",
        "ts115": "TS115.csv",
        "cb513": "CB513.csv",
    }

    for test_name, test_file in test_datasets.items():
        test_dataset = load_dataset("proteinea/SSP", data_files={"test": [test_file]})

        # Save test set to CSV, including both dssp3 and dssp8
        test_df = pd.DataFrame(
            {
                input_column_name: test_dataset["test"][input_column_name],
                dssp3_column_name: test_dataset["test"][dssp3_column_name],
                dssp8_column_name: test_dataset["test"][dssp8_column_name],
                disorder_column_name: test_dataset["test"][disorder_column_name],
            }
        )
        test_df.to_csv(os.path.join(output_dir, f"{test_name}.csv"), index=False)


def combine_and_save_datasets(train_file, val_file, output_file):
    # Load the training and validation datasets
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    # Combine the datasets
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    # Save the combined dataset to a new CSV file
    combined_df.to_csv(output_file, index=False)


# Specify the directory where you want to save the CSV files
output_directory = "./datasets/"

# Fluorescence
load_and_combine_hf_datasets(
    "proteinea/fluorescence",
    "primary",
    "log_fluorescence",
    "primary",
    "log_fluorescence",
    output_dir=output_directory + "fluorescence",
)

# Solubility
load_and_combine_hf_datasets(
    "proteinea/solubility",
    "sequences",
    "labels",
    "sequences",
    "labels",
    output_dir=output_directory + "solubility",
)

# Localization
load_and_combine_hf_datasets(
    "proteinea/deeploc",
    "input",
    "loc",
    "input",
    "loc",
    output_dir=output_directory + "localization",
)

# Call the function to load the datasets and save them as CSV files
load_and_save_secondary_structure_datasets(output_dir=output_directory + "SSP")

# Transporters
combine_and_save_datasets(
    "./datasets/transporters/old/train.csv",
    "./datasets/transporters/old/validation.csv",
    output_file=output_directory + "transporters/train.csv",
)

# membrane proteins (mp)
combine_and_save_datasets(
    "./datasets/mp/old/train.csv",
    "./datasets/mp/old/validation.csv",
    output_file=output_directory + "mp/train.csv",
)
