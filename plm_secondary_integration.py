from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from transformers.trainer_utils import set_seed
import logging
import torch
import sys

from transformers.trainer_callback import EarlyStoppingCallback

from optuna.pruners import MedianPruner


import wandb

import json

import numpy as np


with open('wandb_config.json') as f:
    data = json.load(f)

api_key = data['wandb']['api_key']

wandb_config = {
    "wandb": {
        "project": "Protein-Structure-Prediction",
    }
}

wandb.login(key=api_key)

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Set seed before initializing model.
set_seed(42)

tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-large")

# load the dataset
dataset = load_dataset(
    "proteinea/secondary_structure_prediction",
    data_files={"train": ["training_hhblits.csv"]},
)

# Decide on the number of validation samples
num_validation_samples = 500

# Calculate the number of training samples
num_train_samples = len(dataset["train"]) - num_validation_samples

# Specify the split
train_test_split = dataset["train"].train_test_split(
    test_size=num_validation_samples,
    train_size=num_train_samples,
    seed=42,  # For reproducibility
)

# Access the datasets
train_dataset = train_test_split["train"]
validation_dataset = train_test_split["test"]


# ***************************************************** To be changed *****************************************************
# For debugging purposes, we can use a subset of the training set and validation set
train_dataset = train_dataset.select(range(100))
validation_dataset = validation_dataset.select(range(100))
# ***************************************************** To be changed *****************************************************

input_column_name = "input"
labels_column_name = "dssp3"

# concatenate all sequences
all_sequences = list(train_dataset[input_column_name])

sequence_lengths = [len(seq.split()) for seq in all_sequences]
max_length = int(np.percentile(sequence_lengths, 95))

# Consider each label as a tag for each token
unique_tags = set(
    tag for doc in train_dataset[labels_column_name] for tag in doc
)

# add padding tag
unique_tags.add("<pad>")


tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}


def preprocess_data(examples):
    sequences, labels = examples["input"], examples["dssp3"]

    # remove whitespace and split each sequence into list of amino acids
    sequences = [list("".join(seq.split())) for seq in sequences]
    labels = [list("".join(label.split())) for label in labels]

    # encode sequences
    inputs = tokenizer.batch_encode_plus(
        sequences,
        add_special_tokens=True,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        is_split_into_words=True,
        return_tensors="pt",
    )

    # encode labels
    labels_encoded = [[tag2id[tag] for tag in label] for label in labels]

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
    desc="Running tokenizer on dataset",
)

valid_dataset = validation_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=validation_dataset.column_names,
    desc="Running tokenizer on dataset",
)


def q3_accuracy(y_true, y_pred):
    """
    Computes Q3 accuracy.
    y_true: List of actual values
    y_pred: List of predicted values
    """
    y_true_flat = [
        tag for seq in y_true for tag in seq if tag != tag2id["<pad>"]]
    y_pred_flat = [
        tag for seq in y_pred for tag in seq if tag != tag2id["<pad>"]]
    correct = sum(t == p for t, p in zip(y_true_flat, y_pred_flat))
    return correct / len(y_true_flat)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[0] if isinstance(
        predictions, tuple) else predictions
    # convert numpy ndarray to Tensor
    predictions = torch.tensor(predictions)
    predictions = torch.argmax(predictions, dim=-1)
    return {"q3_accuracy": q3_accuracy(labels.tolist(), predictions.tolist())}


# Prepare the model
def model_init():
    return T5ForConditionalGeneration.from_pretrained("ElnaggarLab/ankh-large")


deepspeed = {
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True
    },
    "wall_clock_breakdown": False
}


training_args = TrainingArguments(
    output_dir="./results",
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_strategy="epoch",
    save_strategy="epoch",
    deepspeed=deepspeed,
    load_best_model_at_end=True,
    metric_for_best_model="eval_q3_accuracy",
    greater_is_better=True,
    max_grad_norm=1.0,
    max_steps=-1,
    logging_steps=500,
    save_steps=500,
    seed=42,
    run_name="SS-Generation",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=64,
    report_to="wandb",
    log_on_each_node=False,
    fp16=True,
)


trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)


def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 10),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-9, 1e-7, log=True),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [1, 2, 4]),
        "per_device_eval_batch_size": trial.suggest_categorical("per_device_eval_batch_size", [2, 4, 8]),
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 1, 64)
    }


def my_hp_name(trial):
    return f"trial_{trial.number}"


# MedianPruner stops the trials whose best intermediate result is worse than median
pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)


# run the hyperparameter search using optuna
best_trial = trainer.hyperparameter_search(
    hp_space=my_hp_space,
    compute_objective=None,
    n_trials=10,
    direction="maximize",
    backend="optuna",
    hp_name=my_hp_name,
    pruner=pruner,
)


# print out the best hyperparameters
print("Best trial hyperparameters:", best_trial.hyperparameters)
