import torch
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import ankh
import wandb
import json
from functools import partial
from sklearn import metrics
from torch.utils.data import Dataset, Subset
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
    EarlyStoppingCallback,
)
from load_embeddings import load_embeddings_and_labels
from classes.ConvBertForMultiClassClassification import (
    ConvBertForMultiClassClassification,
)
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from load_embeddings import load_embeddings_and_labels_combined
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import shap
import matplotlib.pyplot as plt
import random

from sklearn.metrics import confusion_matrix
import seaborn as sns


import logging

# Set up basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Set seed for reproducibility
seed = 7
set_seed(seed)

model_embed_dim = 768


def create_label_mapping(labels):
    unique_labels = set(labels)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    return label_to_int, int_to_label


def transform_labels(labels, label_to_int):
    return [label_to_int[label] for label in labels]


def inverse_transform_labels(int_labels, int_to_label):
    return [int_to_label[i] for i in int_labels]


def load_wandb_config():
    # Load Weights & Biases Configuration
    with open("wandb_config.json") as f:
        data = json.load(f)
    return data["api_key"]


def setup_wandb(api_key):
    # Setup Weights & Biases
    wandb.login(key=api_key)


def load_data(model_type, task):
    # Modify this function to load data based on model and task
    return load_embeddings_and_labels_combined(
        f"./embeddings/{model_type}_{task}", task
    )


class ProteinClassificationDataset(Dataset):
    def __init__(self, sequences, labels, label_to_int=None, membrane=None):
        self.sequences = sequences
        self.labels = labels
        self.label_to_int = label_to_int
        self.membrane = membrane

    def __getitem__(self, idx):
        embedding = self.sequences[idx]
        label = self.labels[idx]

        # If label_to_int mapping exists and label is a string, convert it to an integer
        if self.label_to_int is not None and isinstance(label, str):
            label = self.label_to_int.get(label, -1)  # Default to -1 for unknown labels

        # Handle the case where the label is not found in the mapping
        if label == -1:
            raise ValueError(
                f"Label '{self.labels[idx]}' at index {idx} not found in label mapping."
            )

        if self.label_to_int is None:
            return {
                "embed": torch.tensor(embedding),
                "labels": torch.tensor(label, dtype=torch.float32).unsqueeze(-1),
            }
        else:
            if self.membrane is not None:
                if self.membrane:
                    label = torch.tensor(label, dtype=torch.float).unsqueeze(-1)

                return {
                    "embed": torch.tensor(embedding),
                    "labels": torch.tensor(label),
                }

    def __len__(self):
        return len(self.labels)


def create_datasets(
    training_sequences,
    training_labels,
    test_sequences=None,
    test_labels=None,
    label_to_int=None,
    membrane=None,
):
    # Create the full training dataset
    training_dataset = ProteinClassificationDataset(
        training_sequences, training_labels, label_to_int, membrane
    )
    if test_sequences is not None and test_labels is not None:
        test_dataset = ProteinClassificationDataset(
            test_sequences, test_labels, label_to_int, membrane
        )
        return training_dataset, test_dataset
    return training_dataset


def compute_metrics(p: EvalPrediction, task_type):
    if task_type == "binary":
        preds = (torch.sigmoid(torch.tensor(p.predictions)).numpy() > 0.5).tolist()
        labels = p.label_ids.tolist()
        return {
            "accuracy": metrics.accuracy_score(labels, preds),
            "precision": metrics.precision_score(labels, preds),
            "recall": metrics.recall_score(labels, preds),
            "f1": metrics.f1_score(labels, preds),
            "mcc": metrics.matthews_corrcoef(labels, preds),
        }

    elif task_type == "multiclass":
        # Use softmax and argmax for multiclass predictions
        preds = torch.argmax(
            torch.softmax(torch.tensor(p.predictions), dim=1), dim=1
        ).numpy()
        labels = p.label_ids
        return {
            "accuracy": metrics.accuracy_score(labels, preds),
            # Use 'macro' or 'weighted' average in multiclass metrics
            "precision": metrics.precision_score(labels, preds, average="weighted"),
            "recall": metrics.recall_score(labels, preds, average="weighted"),
            "f1": metrics.f1_score(labels, preds, average="weighted"),
            "mcc": metrics.matthews_corrcoef(labels, preds),
        }

    elif task_type == "regression":
        return {
            "spearmanr": stats.spearmanr(p.label_ids, p.predictions).correlation,
        }


def model_init(embed_dim, task_type=None, num_classes=None, training_labels_mean=None):
    hidden_dim = int(embed_dim / 2)
    num_hidden_layers = 1
    nlayers = 1
    nhead = 4
    dropout = 0.2
    conv_kernel_size = 7
    pooling = "max"
    if task_type == "multiclass":
        downstream_model = ConvBertForMultiClassClassification(
            input_dim=embed_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_layers=nlayers,
            kernel_size=conv_kernel_size,
            dropout=dropout,
            pooling=pooling,
            num_classes=num_classes,
        )
    elif task_type == "binary":
        downstream_model = ankh.ConvBertForBinaryClassification(
            input_dim=embed_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_layers=nlayers,
            kernel_size=conv_kernel_size,
            dropout=dropout,
            pooling=pooling,
        )
    elif task_type == "regression":
        downstream_model = ankh.ConvBertForRegression(
            input_dim=embed_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_layers=nlayers,
            kernel_size=conv_kernel_size,
            dropout=dropout,
            pooling=pooling,
            training_labels_mean=training_labels_mean,
        )
    return downstream_model


def load_sequences(task):
    # Load the amino acid sequences from the corresponding CSV file
    sequences_df = pd.read_csv(f"./datasets/{task}/test.csv")
    return sequences_df


def max_pooling(embeddings):
    # Apply max pooling to each embedding in the list
    pooled_embeddings = [np.max(embedding, axis=0) for embedding in embeddings]
    return pooled_embeddings


def capture_classifications(p: EvalPrediction, task_type):
    if task_type == "binary":
        preds = (torch.sigmoid(torch.tensor(p.predictions)).numpy() > 0.5).astype(int)
    elif task_type == "multiclass":
        preds = torch.argmax(
            torch.softmax(torch.tensor(p.predictions), dim=1), dim=1
        ).numpy()

    labels = p.label_ids
    correct_indices = np.where(preds == labels)[0]
    return correct_indices, preds, labels


def find_common_correctly_classified(samples_dict):
    correctly_classified_ankh = samples_dict["ankh"]
    correctly_classified_p2s = samples_dict["p2s"]

    common_samples = []
    for index in correctly_classified_ankh:
        if index in correctly_classified_p2s:
            common_samples.append(index)

    return common_samples


def save_common_correctly_classified_sequences(indices, task, sequences_df):
    if task in ["ionchannels", "transporters", "mp"]:
        col_name = "sequence"
    elif task == "localization":
        col_name = "input"
    elif task == "solubility":
        col_name = "sequences"
    # Extract sequences using indices from common correctly classified samples
    correctly_classified_sequences = sequences_df.iloc[indices][[col_name]]

    correctly_classified_sequences = correctly_classified_sequences.rename(
        columns={col_name: "sequence"}
    )

    # Save to CSV
    correctly_classified_sequences.to_csv(
        f"./correctly_classified_sequences/{task}_common_correctly_classified.csv",
        index=False,
    )
    print(f"Saved common correctly classified sequences for task: {task}")


def main():
    # create a new folder to store results
    os.makedirs("correctly_classified_sequences", exist_ok=True)

    # Load Weights & Biases API key
    api_key = load_wandb_config()
    setup_wandb(api_key)

    results = []
    models = ["p2s", "ankh"]
    tasks = [
        "transporters",
        "localization",
        "solubility",
        "ionchannels",
        "mp",
    ]
    membrane_tasks = ["transporters", "ionchannels", "mp"]

    p2s_hyperparams = {
        "solubility": {
            "learning_rate": 9.238025319681167e-05,
            "weight_decay": 0.1,
            "warmup_ratio": 0.1,
            "gradient_accumulation_steps": 12,
        },
        "ssp": {
            "learning_rate": 0.0030620212672418912,
            "weight_decay": 0.08,
            "warmup_ratio": 0.35000000000000003,
            "gradient_accumulation_steps": 7,
        },
        "fluorescence": {
            "learning_rate": 0.00011341216929700865,
            "weight_decay": 0.07,
            "warmup_ratio": 0.15000000000000002,
            "gradient_accumulation_steps": 10,
        },
        "localization": {
            "learning_rate": 0.000570396865105337,
            "weight_decay": 0.1,
            "warmup_ratio": 0.30000000000000004,
            "gradient_accumulation_steps": 14,
        },
        "ionchannels": {
            "learning_rate": 0.0003322487943750368,
            "weight_decay": 0.1,
            "warmup_ratio": 0.05,
            "gradient_accumulation_steps": 15,
        },
        "mp": {
            "learning_rate": 0.00012531554540856643,
            "weight_decay": 0.05,
            "warmup_ratio": 0.25,
            "gradient_accumulation_steps": 16,
        },
        "transporters": {
            "learning_rate": 0.0017708480428902313,
            "weight_decay": 0.07,
            "warmup_ratio": 0.45,
            "gradient_accumulation_steps": 6,
        },
    }

    ankh_learning_rate = 1e-03
    ankh_warmup_steps = 1000
    ankh_gradient_accumulation_steps = 16
    ankh_weight_decay = 0.0

    for task in tasks:
        # Dictionaries to hold misclassified samples for each model
        correctly_classified_samples_model = {}
        for model in models:
            print("****************************************************************")
            print(f"Training on the {task} task and {model} model.")
            print("****************************************************************")
            # set if the task is binary, multiclass, or regression
            if task == "localization":
                task_type = "multiclass"
            elif task == "fluorescence":
                task_type = "regression"
            else:
                task_type = "binary"

            def capture_eval_pred(trainer, eval_dataset):
                predictions, label_ids, metrics = trainer.predict(eval_dataset)
                if task_type == "binary":
                    preds = (
                        torch.sigmoid(torch.tensor(predictions)).numpy() > 0.5
                    ).astype(int)
                elif task_type == "multiclass":
                    preds = torch.argmax(
                        torch.softmax(torch.tensor(predictions), dim=1), dim=1
                    ).numpy()
                else:
                    preds = predictions  # For regression tasks
                return preds, label_ids, metrics

            experiment = f"{task}_best_{model}"
            # Load data for the current model and task
            train_embeddings, train_labels, test_embeddings, test_labels = load_data(
                model, task
            )

            label_to_int, int_to_label = None, None
            if task in ["localization", "ionchannels", "mp", "transporters"]:
                label_to_int, int_to_label = create_label_mapping(
                    set(train_labels + test_labels)
                )
                train_labels = transform_labels(train_labels, label_to_int)
                test_labels = transform_labels(test_labels, label_to_int)

            training_labels_mean = None
            if task == "fluorescence":
                training_labels_mean = np.mean(train_labels)

            num_classes = None
            if task == "localization":
                num_classes = len(np.unique(train_labels))

            if model == "p2s":
                hyperparams = p2s_hyperparams[task]
                # Use hyperparams['learning_rate'], hyperparams['weight_decay'], etc.
            elif model == "ankh":
                hyperparams = {
                    "learning_rate": ankh_learning_rate,
                    "warmup_steps": ankh_warmup_steps,
                    "gradient_accumulation_steps": ankh_gradient_accumulation_steps,
                    "weight_decay": ankh_weight_decay,
                }
                # Use ankh_learning_rate, ankh_warmup_steps, etc.

            indices = np.arange(len(train_embeddings))
            if task_type in ["binary", "multiclass"]:
                # Use custom_stratified_split for classification tasks
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                cv_splitter = skf.split
            elif task_type == "regression":
                # Use KFold for regression tasks
                kf = KFold(n_splits=5, shuffle=True, random_state=seed)
                cv_splitter = kf.split

            training_args = TrainingArguments(
                output_dir=f"./results_{experiment}",
                warmup_steps=hyperparams["warmup_steps"] if model == "ankh" else 0,
                warmup_ratio=hyperparams["warmup_ratio"] if model == "p2s" else 0,
                learning_rate=hyperparams["learning_rate"],
                weight_decay=hyperparams["weight_decay"],
                gradient_accumulation_steps=hyperparams["gradient_accumulation_steps"],
                num_train_epochs=10,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                logging_dir=f"./logs_{experiment}",
                logging_steps=500,
                do_train=True,
                do_eval=True,
                evaluation_strategy="epoch",
                fp16=False,
                fp16_opt_level="O2",
                run_name=experiment,
                load_best_model_at_end=True,
                metric_for_best_model="eval_spearmanr"
                if task_type == "regression"
                else "eval_mcc",
                greater_is_better=True,
                save_strategy="epoch",
                report_to="wandb",
            )

            final_trainer = Trainer(
                model_init=partial(
                    model_init,
                    embed_dim=model_embed_dim,
                    task_type=task_type,
                    num_classes=num_classes,
                    training_labels_mean=training_labels_mean,
                ),
                args=training_args,
                train_dataset=create_datasets(
                    train_embeddings,
                    train_labels,
                    label_to_int=label_to_int,
                    membrane=task in membrane_tasks,
                ),
                eval_dataset=create_datasets(
                    test_embeddings,
                    test_labels,
                    label_to_int=label_to_int,
                    membrane=task in membrane_tasks,
                ),
                compute_metrics=partial(compute_metrics, task_type=task_type),
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=5, early_stopping_threshold=0.05
                    )
                ],
            )

            print(
                f"Training on the full training set for the {task} task and {model} model."
            )

            # Train the model on the full training set
            final_trainer.train()

            # After cross-validation, evaluate on the test set
            test_dataset = create_datasets(
                test_embeddings,
                test_labels,
                label_to_int=label_to_int,
                membrane=task in membrane_tasks,
            )

            # Perform prediction on the test dataset
            predictions, label_ids, metrics = final_trainer.predict(test_dataset)

            correct_indices, preds, labels = capture_classifications(
                EvalPrediction(predictions=predictions, label_ids=label_ids),
                task_type,
            )
            correctly_classified_samples_model[model] = correct_indices

        # Find common correctly classified samples
        common_correctly_classified_samples = find_common_correctly_classified(
            correctly_classified_samples_model
        )

        # Load sequences for saving misclassified ones
        sequences_df = load_sequences(task)

        # Save the common misclassified sequences
        save_common_correctly_classified_sequences(
            common_correctly_classified_samples, task, sequences_df
        )


if __name__ == "__main__":
    main()
