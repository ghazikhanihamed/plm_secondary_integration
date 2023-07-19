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

from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

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

wandb.login(key=api_key, force=True, relogin=True)

wandb.init()

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
training_dataset = load_dataset(
    "proteinea/secondary_structure_prediction",
    data_files={"train": ["training_hhblits.csv"]},
)

# we randomly select 500 samples from the training set to use as our validation set
validation_dataset = training_dataset["train"].train_test_split(
    test_size=500, seed=42)

input_column_name = "input"
labels_column_name = "dssp3"

# concatenate all sequences
all_sequences = list(training_dataset["train"][input_column_name])

sequence_lengths = [len(seq.split()) for seq in all_sequences]
max_length = int(np.percentile(sequence_lengths, 95))

# Consider each label as a tag for each token
unique_tags = set(
    tag for doc in training_dataset["train"][labels_column_name] for tag in doc
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


train_dataset = training_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=training_dataset.column_names["train"],
    desc="Running tokenizer on dataset",
)

valid_dataset = validation_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=validation_dataset.column_names["train"],
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
    "fp16": {
        "enabled": "auto"
    },
    "optimizer": {
        "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "weight_decay": "auto",
                    "torch_adam": True,
                    "adam_w_mode": True
                }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": "auto",
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto",
                    "total_num_steps": "auto"
                }
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}

training_args = TrainingArguments(
    output_dir="./results",
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    logging_dir="./logs",
    logging_strategy="steps",
    save_strategy="steps",
    deepspeed=deepspeed,
    load_best_model_at_end=True,
    metric_for_best_model="eval_q3_accuracy",
    greater_is_better=True,
    num_train_epochs=20,
    fp16=False,
    max_grad_norm=1.0,
    max_steps=-1,
    logging_steps=500,
    save_steps=500,
    seed=42,
    run_name="SS-Generation",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    log_level="info",
    report_to="wandb",
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset["train"],
    eval_dataset=valid_dataset["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# define the hyperparameters search space
config = {
    "learning_rate": tune.loguniform(1e-6, 1e-2),
    "weight_decay": tune.loguniform(1e-6, 1e-4),
}

# configure the resources per trial for the GPUs
resources_per_trial = {"gpu": 4}

# define the reporter to fetch the important information
reporter = CLIReporter(
    parameter_columns=["learning_rate", "weight_decay"],
    metric_columns=["loss", "q3_accuracy", "training_iteration"],
)

best_trial = trainer.hyperparameter_search(
    hp_space=lambda _: config,
    backend="ray",
    n_trials=10,
    search_alg=HyperOptSearch(metric="eval_q3_accuracy", mode="max"),
    scheduler=ASHAScheduler(metric="eval_q3_accuracy", mode="max"),
    kwargs={"resources_per_trial": resources_per_trial},
)

# print out the best hyperparameters
print("Best trial hyperparameters:", best_trial.hyperparameters)
