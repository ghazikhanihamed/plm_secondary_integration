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


import logging

# Set up basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Set seed for reproducibility
seed = 7
set_seed(seed)

model_embed_dim = 768


def custom_stratified_split(indices, labels, n_splits=5):
    folds = []
    remaining_indices = indices.copy()

    for _ in range(n_splits):
        # Stratified split for each fold
        train_idx, val_idx = train_test_split(
            remaining_indices,
            test_size=1.0 / n_splits,
            stratify=[labels[i] for i in remaining_indices],
        )

        folds.append((train_idx, val_idx))
        remaining_indices = [idx for idx in remaining_indices if idx not in val_idx]

    return folds


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
    def __init__(self, sequences, labels, label_encoder=None, membrane=None):
        self.sequences = sequences
        self.labels = labels
        self.label_encoder = label_encoder
        self.membrane = membrane

    def __getitem__(self, idx):
        try:
            embedding = self.sequences[idx]
            label = self.labels[idx]

            # Decode byte string to regular string if necessary
            if isinstance(label, bytes):
                label = label.decode("utf-8")

            # Label encoding
            if self.label_encoder is not None:
                label = self.label_encoder.transform([label])[0]

            # Tensor conversion
            label_tensor = torch.tensor(
                label, dtype=torch.float32 if self.membrane is None else torch.long
            )

            return {
                "embed": torch.tensor(embedding),
                "labels": label_tensor.unsqueeze(-1),
            }

            # if self.label_encoder is None:
            #     return {
            #         "embed": torch.tensor(embedding),
            #         "labels": torch.tensor(label, dtype=torch.float32).unsqueeze(-1),
            #     }
            # else:
            #     # Convert string label to integer
            #     label_int = self.label_encoder.transform([label])[0]
            #     if self.membrane is not None:
            #         if self.membrane:
            #             label_int = torch.tensor(
            #                 label_int, dtype=torch.float
            #             ).unsqueeze(-1)

            #         return {
            #             "embed": torch.tensor(embedding),
            #             "labels": torch.tensor(label_int),
            #         }
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            print(f"Label causing issue: {self.labels[idx]}")
            print(f"label type: {type(self.labels[idx])}")
            raise e

    def __len__(self):
        return len(self.labels)


def create_datasets(
    training_sequences,
    training_labels,
    test_sequences=None,
    test_labels=None,
    label_encoder=None,
    membrane=None,
):
    # Create the full training dataset
    training_dataset = ProteinClassificationDataset(
        training_sequences, training_labels, label_encoder, membrane
    )
    if test_sequences is not None and test_labels is not None:
        test_dataset = ProteinClassificationDataset(
            test_sequences, test_labels, label_encoder, membrane
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


def main():
    # create a new folder to store results
    os.makedirs("results", exist_ok=True)
    # Load Weights & Biases API key
    api_key = load_wandb_config()
    setup_wandb(api_key)

    results = []
    models = ["p2s", "ankh"]
    tasks = [
        "localization",
        "fluorescence",
        "solubility",
        "transporters",
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

    for model in models:
        for task in tasks:
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

            experiment = f"{task}_best_{model}"
            # Load data for the current model and task
            train_embeddings, train_labels, test_embeddings, test_labels = load_data(
                model, task
            )

            label_encoder = None
            if task in ["localization", "ionchannels", "mp", "transporters"]:
                label_encoder = LabelEncoder()
                label_encoder.fit(list(set(train_labels + test_labels)))
                train_labels_encoded = label_encoder.transform(train_labels)

            training_labels_mean = None
            if task == "fluorescence":
                training_labels_mean = np.mean(train_labels)

            num_classes = None
            if task == "localization":
                num_classes = len(np.unique(train_labels_encoded))

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

            for fold, (train_idx, val_idx) in enumerate(
                cv_splitter(indices, train_labels)
            ):
                print(
                    "****************************************************************"
                )
                print(f"Training on fold {fold} of the {task} task and {model} model.")
                print(
                    "****************************************************************"
                )
                # Split data into training and validation for the current fold
                training_sequences = [train_embeddings[i] for i in train_idx]
                training_labels = [train_labels[i] for i in train_idx]
                validation_sequences = [train_embeddings[i] for i in val_idx]
                validation_labels = [train_labels[i] for i in val_idx]

                # Create the training and validation datasets
                training_dataset, validation_dataset = create_datasets(
                    training_sequences,
                    training_labels,
                    validation_sequences,
                    validation_labels,
                    label_encoder,
                    membrane=task in membrane_tasks,
                )

                training_args = TrainingArguments(
                    output_dir=f"./results_{experiment}",
                    warmup_steps=hyperparams["warmup_steps"] if model == "ankh" else 0,
                    warmup_ratio=hyperparams["warmup_ratio"] if model == "p2s" else 0,
                    learning_rate=hyperparams["learning_rate"],
                    weight_decay=hyperparams["weight_decay"],
                    gradient_accumulation_steps=hyperparams[
                        "gradient_accumulation_steps"
                    ],
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

                trainer = Trainer(
                    model_init=partial(
                        model_init,
                        embed_dim=model_embed_dim,
                        task_type=task_type,
                        num_classes=num_classes,
                        training_labels_mean=training_labels_mean,
                    ),
                    args=training_args,
                    train_dataset=training_dataset,
                    eval_dataset=validation_dataset,
                    compute_metrics=partial(compute_metrics, task_type=task_type),
                    callbacks=[
                        EarlyStoppingCallback(
                            early_stopping_patience=5, early_stopping_threshold=0.05
                        )
                    ],
                )

                # Train the model
                trainer.train()

                # Evaluate the model on the validation set
                eval_results = trainer.evaluate(validation_dataset)

                # Store cross-validation results
                for key in eval_results.keys():
                    results.append(
                        {
                            "model": model,
                            "task": task,
                            "fold": fold,
                            "metric": key,  # e.g., "accuracy", "precision", etc.
                            "value": eval_results[key],  # The value of the metric
                            "data_split": "cross-validation",
                        }
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
                    label_encoder=label_encoder,
                    membrane=task in membrane_tasks,
                ),
                eval_dataset=create_datasets(
                    test_embeddings,
                    test_labels,
                    label_encoder=label_encoder,
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
                label_encoder,
                membrane=task in membrane_tasks,
            )
            test_results = final_trainer.evaluate(test_dataset)

            # Store test set results
            for key in test_results.keys():
                results.append(
                    {
                        "model": model,
                        "task": task,
                        "metric": key,  # e.g., "accuracy", "precision", etc.
                        "value": test_results[key],  # The value of the metric
                        "data_split": "test",
                    }
                )

    # Convert results to DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    # Save results to CSV in the results folder
    results_df.to_csv(f"./results/cross_validation_results.csv", index=False)


if __name__ == "__main__":
    main()
