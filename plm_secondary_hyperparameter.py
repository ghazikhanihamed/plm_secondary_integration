import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, concatenate_datasets
import torch
import numpy as np

import wandb
import json
from accelerate.utils import set_seed

set_seed(7)

# Load Weights & Biases Configuration
with open("wandb_config.json") as f:
    data = json.load(f)

api_key = data["api_key"]

wandb_config = {
    "project": "plm_secondary_integration",
}

wandb.login(key=api_key)

tokenizer = T5TokenizerFast.from_pretrained("ElnaggarLab/ankh-base", use_fast=True)

# Load the datasets
dataset1 = load_dataset(
    "proteinea/secondary_structure_prediction",
    data_files={"train": ["training_hhblits.csv"]},
)
dataset2 = load_dataset(
    "proteinea/secondary_structure_prediction",
    data_files={"CASP12": ["CASP12.csv"]},
)
dataset3 = load_dataset(
    "proteinea/secondary_structure_prediction",
    data_files={"CASP14": ["CASP14.csv"]},
)
dataset4 = load_dataset(
    "proteinea/secondary_structure_prediction", data_files={"CB513": ["CB513.csv"]}
)
dataset5 = load_dataset(
    "proteinea/secondary_structure_prediction", data_files={"TS115": ["TS115.csv"]}
)

# Keep dataset1 as the training set
train_dataset = dataset1["train"]

# validation set
validation_dataset = concatenate_datasets(
    [dataset2["CASP12"], dataset3["CASP14"], dataset4["CB513"], dataset5["TS115"]]
)

input_column_name = "input"
labels_column_name = "dssp3"

# concatenate all sequences
all_sequences = list(train_dataset[input_column_name])
sequence_lengths = [len(seq) for seq in all_sequences]
max_length = int(np.percentile(sequence_lengths, 99))

# Consider each label as a tag for each token
unique_tags = set(tag for doc in train_dataset[labels_column_name] for tag in doc)

# add padding tag
unique_tags.add("<pad>")


tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}


def preprocess_data(examples):
    sequences, labels = examples["input"], examples["dssp3"]

    # remove whitespace and split each sequence into list of amino acids
    sequences = [list("".join(seq.split())) for seq in sequences]
    labels = [list("".join(label.split())) for label in labels]

    # print("Max length: ", max_length)

    # encode sequences
    inputs = tokenizer(
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
    desc="Running tokenizer on dataset for training",
)

valid_dataset = validation_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=validation_dataset.column_names,
    desc="Running tokenizer on dataset for validation",
)


def q3_accuracy(y_true, y_pred):
    """
    Computes Q3 accuracy.
    y_true: List of actual values
    y_pred: List of predicted values
    """
    y_true_flat = [tag for seq in y_true for tag in seq if tag != tag2id["<pad>"]]
    y_pred_flat = [tag for seq in y_pred for tag in seq if tag != tag2id["<pad>"]]
    correct = sum(t == p for t, p in zip(y_true_flat, y_pred_flat))
    return correct / len(y_true_flat)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[0] if isinstance(predictions, tuple) else predictions
    # convert numpy ndarray to Tensor
    predictions = torch.tensor(predictions)
    predictions = torch.argmax(predictions, dim=-1)
    return {"q3_accuracy": q3_accuracy(labels.tolist(), predictions.tolist())}


experiment = "p2s_hyperparameters"


def optuna_hp_space(trial):
    return {
        # Learning rate: For fine-tuning, a smaller range focused on smaller values is preferred.
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        # Weight decay: This range is reasonable for fine-tuning, but still not too high.
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1, step=0.01),
        # Number of epochs: For fine-tuning, you might not need as many epochs, especially with a smaller dataset.
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        # Warmup ratio: As before, a step is introduced to not have a very granular search in this space.
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.5, step=0.05),
        # Gradient accumulation steps: Since the dataset is not huge, fewer accumulation steps might suffice.
        "gradient_accumulation_steps": trial.suggest_int(
            "gradient_accumulation_steps", 1, 4
        ),
    }


def model_init():
    return T5ForConditionalGeneration.from_pretrained("ElnaggarLab/ankh-base")


def compute_objective(metrics):
    # This function will compute the objective for optimization during hyperparameter tuning.
    # Here, we aim to maximize Q3 accuracy.
    return metrics["eval_q3_accuracy"]


# Prepare training args
training_args = TrainingArguments(
    output_dir=f"./results_{experiment}",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    warmup_ratio=0.3,
    learning_rate=1e-3,
    logging_dir=f"./logs_{experiment}",
    logging_steps=50,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=1,
    fp16=False,
    fp16_opt_level="O2",
    load_best_model_at_end=True,
    metric_for_best_model="eval_q3_accuracy",
    greater_is_better=True,
    save_strategy="epoch",
    remove_unused_columns=False,
    run_name="SS-Generation",
    report_to="wandb",
)

# Initialize Trainer
trainer = Trainer(
    model=None,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    model_init=model_init,
)

best_trials = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=20,
    compute_objective=compute_objective,
)

print("------------------")
print(best_trials.hyperparameters)
print("------------------")
