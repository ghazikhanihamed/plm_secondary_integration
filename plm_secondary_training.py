from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, concatenate_datasets
import logging
import torch
import numpy as np
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import wandb
import json
import random

# Load Weights & Biases Configuration
with open("wandb_config.json") as f:
    data = json.load(f)

api_key = data["api_key"]

wandb_config = {
    "project": "plm_secondary_integration",
}

wandb.login(key=api_key)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base", use_fast=True)

# Load the datasets
dataset1 = load_dataset(
    "proteinea/secondary_structure_prediction",
    data_files={"train": ["training_hhblits.csv"]},
)["train"]
dataset2 = load_dataset(
    "proteinea/secondary_structure_prediction", data_files={"CASP12": ["CASP12.csv"]}
)["CASP12"]
dataset3 = load_dataset(
    "proteinea/secondary_structure_prediction", data_files={"CASP14": ["CASP14.csv"]}
)["CASP14"]
dataset4 = load_dataset(
    "proteinea/secondary_structure_prediction", data_files={"CB513": ["CB513.csv"]}
)["CB513"]
dataset5 = load_dataset(
    "proteinea/secondary_structure_prediction", data_files={"TS115": ["TS115.csv"]}
)["TS115"]

# Concatenate all datasets
all_datasets = concatenate_datasets([dataset1, dataset2, dataset3, dataset4, dataset5])

# Split the concatenated dataset into training and validation sets
splits = all_datasets.train_test_split(test_size=0.1, seed=42)

train_dataset = splits["train"]
validation_dataset = splits["test"]

input_column_name = "dssp3"
labels_column_name = "dssp3"

max_length = 512

# Consider each label as a tag for each token
unique_tags = set(tag for doc in train_dataset[labels_column_name] for tag in doc)

# add padding and masking tag
unique_tags.add("<pad>")
unique_tags.add("<extra_id_0>")


tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}


def mask_secondary_structure(sequences, mask_prob=0.20, mask_token="<extra_id_0>"):
    masked_sequences = []
    for seq in sequences:  # seq is a list of amino acids
        masked_seq = []
        for element in seq:
            if random.random() < mask_prob:
                masked_seq.append(mask_token)
            else:
                masked_seq.append(element)
        masked_sequences.append(masked_seq)
    return masked_sequences


def preprocess_data(examples):
    # Now focusing on secondary structure
    sequences = examples["dssp3"]

    # remove whitespace and split each sequence into list of secondary structure elements
    sequences = [list("".join(seq.split())) for seq in sequences]

    # Apply masking to sequences (20% mask probability by default)
    masked_sequences = mask_secondary_structure(sequences)

    # encode sequences
    inputs = tokenizer(
        masked_sequences,
        add_special_tokens=True,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        is_split_into_words=True,
        return_tensors="pt",
    )

    # This part remains same as labels now point to secondary structure elements
    labels_encoded = [[tag2id[tag] for tag in seq] for seq in masked_sequences]

    # Pad or truncate the labels to match the sequence length
    labels_encoded = [
        label[:max_length] + [tag2id["<pad>"]] * (max_length - len(label))
        for label in labels_encoded
    ]

    assert len(inputs["input_ids"]) == len(labels_encoded)

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": torch.tensor(labels_encoded, dtype=torch.long),
    }


train_dataset = train_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Running tokenizer on dataset for training",
)

valid_dataset = validation_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=validation_dataset.column_names,
    desc="Running tokenizer on dataset for validation",
)


def compute_metrics(p):
    # Flatten the predictions and labels, and remove padding tags
    predictions = np.argmax(p.predictions, axis=2).flatten()
    labels = p.label_ids.flatten()

    # Remove padding (<pad>) token id, which is 0 in this case
    mask = labels != tag2id["<pad>"]
    predictions = predictions[mask]
    labels = labels[mask]

    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    f1 = f1_score(labels, predictions, average="weighted")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Create the model and prepare it
model = T5ForConditionalGeneration.from_pretrained("ElnaggarLab/ankh-base")

experiment = "secondary_integration"

# Compute total steps
num_train_samples = len(train_dataset)
batch_size = 4  # Replace with your actual batch size
num_epochs = 50  # Replace with your actual number of epochs
total_steps = (num_train_samples // batch_size) * num_epochs

# Compute warmup_steps
warmup_steps = int(0.1 * total_steps)

# Prepare training args
training_args = TrainingArguments(
    output_dir=f"./results_{experiment}",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    learning_rate=4e-3,
    logging_dir=f"./logs_{experiment}",
    logging_steps=200,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=32,
    fp16=False,
    fp16_opt_level="O2",
    seed=42,
    save_strategy="epoch",
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    run_name="secondary_integration",
    report_to="wandb",
    hub_token="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",
    hub_model_id="ghazikhanihamed/secondary_integration",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# We save the best model in the folder "best_model"
trainer.save_model(f"./best_model_{experiment}")

# Push model to hub
trainer.push_to_hub(commit_message="PLM-Secondary-Integration")
