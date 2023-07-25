import gc
import torch
from optuna.integration.wandb import WeightsAndBiasesCallback

import optuna
from optuna.samplers import RandomSampler

from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, concatenate_datasets
from transformers.trainer_utils import set_seed
import logging
import sys

import wandb
import json
import numpy as np

# Load Weights & Biases Configuration
with open('wandb_config.json') as f:
    data = json.load(f)

api_key = data['wandb']['api_key']

wandb_config = {
    "project": "Protein-Structure-Prediction",
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
dataset1 = load_dataset(
    "proteinea/secondary_structure_prediction",
    data_files={"train": ["training_hhblits.csv"]},
)
dataset2 = load_dataset(
    "proteinea/secondary_structure_prediction", data_files={'CASP12': ['CASP12.csv']})
dataset3 = load_dataset(
    "proteinea/secondary_structure_prediction", data_files={'CASP14': ['CASP14.csv']})
dataset4 = load_dataset(
    "proteinea/secondary_structure_prediction", data_files={'CB513': ['CB513.csv']})
dataset5 = load_dataset(
    "proteinea/secondary_structure_prediction", data_files={'TS115': ['TS115.csv']})

# concatenate dataset1 and dataset4
train_dataset = concatenate_datasets(
    [dataset1["train"], dataset4["CB513"]])

# The validation set will be dataset5
validation_dataset = dataset5["TS115"]

# Two separate test datasets
test_dataset1 = dataset2["CASP12"]
test_dataset2 = dataset3["CASP14"]

# Print the number of samples
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(validation_dataset)}")
print(f"Number of test samples on CASP12: {len(test_dataset1)}")
print(f"Number of test samples on CASP14: {len(test_dataset2)}")


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
    desc="Running tokenizer on dataset for training",
)

valid_dataset = validation_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=validation_dataset.column_names,
    desc="Running tokenizer on dataset for validation",
)

test_dataset1 = test_dataset1.map(
    preprocess_data,
    batched=True,
    remove_columns=test_dataset1.column_names,
    desc="Running tokenizer on dataset for test",
)

test_dataset2 = test_dataset2.map(
    preprocess_data,
    batched=True,
    remove_columns=test_dataset2.column_names,
    desc="Running tokenizer on dataset for test",
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
def model_init(trial):
    return T5ForConditionalGeneration.from_pretrained("ElnaggarLab/ankh-large")


deepspeed = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True
    },
    "gradient_clipping": "auto",
    "gradient_accumulation_steps": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}


def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)

    # Create a new training_args and use the hyperparameters
    training_args = TrainingArguments(
        output_dir="./results",
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_strategy="epoch",
        deepspeed=deepspeed,
        metric_for_best_model="eval_q3_accuracy",
        greater_is_better=True,
        logging_steps=500,
        seed=42,
        run_name="SS-Generation",
        report_to="wandb",
        learning_rate=learning_rate,
        num_train_epochs=1,
    )

    # Create a new Trainer instance
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    metrics = trainer.evaluate()

    # Return the evaluation metric for the trial
    return metrics["eval_q3_accuracy"]


def clear_cuda_cache(study, trial):
    torch.cuda.empty_cache()


if __name__ == "__main__":
    n_trials = 10

    wandbc = WeightsAndBiasesCallback(
        metric_name="eval_q3_accuracy", wandb_kwargs=wandb_config)

    study = optuna.create_study(
        direction="maximize", sampler=RandomSampler())
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, callbacks=[
                   wandbc, clear_cuda_cache])

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Retrain model with best hyperparameters
    best_params = study.best_params

    # Create a new training_args and use the hyperparameters
    re_training_args = TrainingArguments(
        output_dir="./results",
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_strategy="epoch",
        deepspeed=deepspeed,
        metric_for_best_model="eval_q3_accuracy",
        greater_is_better=True,
        logging_steps=500,
        seed=42,
        run_name="SS-Generation",
        report_to="wandb",
        learning_rate=best_params['learning_rate'],
        num_train_epochs=20,
    )

    trainer = Trainer(
        model_init=model_init,
        args=re_training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Evaluate the model on test datasets
    metrics_test1 = trainer.evaluate(test_dataset=test_dataset1)
    metrics_test2 = trainer.evaluate(test_dataset=test_dataset2)

    print("Evaluation results on test set CASP12: ", metrics_test1)
    print("Evaluation results on test set CASP14: ", metrics_test2)
