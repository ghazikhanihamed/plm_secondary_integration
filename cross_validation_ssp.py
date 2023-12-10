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
from load_embeddings import load_ssp_embeddings_and_labels, load_embeddings_and_labels
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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


def load_wandb_config():
    # Load Weights & Biases Configuration
    with open("wandb_config.json") as f:
        data = json.load(f)
    return data["api_key"]


def setup_wandb(api_key):
    # Setup Weights & Biases
    wandb.login(key=api_key)


def load_data():
    # Modify this function to load data based on model and task
    return load_ssp_embeddings_and_labels("./embeddings/p2s_SSP")


class SSPDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        embedding = self.encodings[idx]
        labels = self.labels[idx]
        return {"embed": torch.tensor(embedding), "labels": torch.tensor(labels)}

    def __len__(self):
        return len(self.labels)


def create_datasets(
    training_sequences,
    training_labels,
    casp12_sequences,
    casp12_labels,
    casp13_sequences,
    casp13_labels,
    casp14_sequences,
    casp14_labels,
    ts115_sequences,
    ts115_labels,
    cb513_sequences,
    cb513_labels,
):
    training_dataset = SSPDataset(training_sequences, training_labels)
    casp12_dataset = SSPDataset(casp12_sequences, casp12_labels)
    casp13_dataset = SSPDataset(casp13_sequences, casp13_labels)
    casp14_dataset = SSPDataset(casp14_sequences, casp14_labels)
    ts115_dataset = SSPDataset(ts115_sequences, ts115_labels)
    cb513_dataset = SSPDataset(cb513_sequences, cb513_labels)

    return (
        training_dataset,
        casp12_dataset,
        casp13_dataset,
        casp14_dataset,
        ts115_dataset,
        cb513_dataset,
    )


def create_datasets_ssp(
    training_sequences,
    training_labels,
    validation_sequences=None,
    validation_labels=None,
):
    training_dataset = SSPDataset(training_sequences, training_labels)
    if validation_sequences is not None and validation_labels is not None:
        validation_dataset = SSPDataset(validation_sequences, validation_labels)
        return (
            training_dataset,
            validation_dataset,
        )
    else:
        return training_dataset


def mask_disorder(labels, masks):
    for label, mask in zip(labels, masks):
        for i, disorder in enumerate(mask):
            if disorder == "0.0":
                label[i] = -100
    return labels


def model_init(num_tokens, embed_dim):
    hidden_dim = int(embed_dim / 2)
    num_hidden_layers = 1  # Number of hidden layers in ConvBert.
    nlayers = 1  # Number of ConvBert layers.
    nhead = 4
    dropout = 0.2
    conv_kernel_size = 7
    downstream_model = ankh.ConvBertForMultiClassClassification(
        num_tokens=num_tokens,
        input_dim=embed_dim,
        nhead=nhead,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        num_layers=nlayers,
        kernel_size=conv_kernel_size,
        dropout=dropout,
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
        "SSP",
    ]
    ssp_types = ["ssp3", "ssp8"]
    kf = KFold(
        n_splits=5, shuffle=True, random_state=seed
    )  # Use KFold instead of StratifiedKFold

    p2s_hyperparams = {
        "learning_rate": 0.0030620212672418912,
        "weight_decay": 0.08,
        "warmup_ratio": 0.35000000000000003,
        "gradient_accumulation_steps": 7,
    }

    ankh_learning_rate = 1e-03
    ankh_warmup_steps = 1000
    ankh_gradient_accumulation_steps = 16
    ankh_weight_decay = 0.0

    task = "SSP"

    for model in models:
        for ssp in ssp_types:
            if model == "p2s":
                hyperparams = p2s_hyperparams
            elif model == "ankh":
                hyperparams = {
                    "learning_rate": ankh_learning_rate,
                    "warmup_steps": ankh_warmup_steps,
                    "gradient_accumulation_steps": ankh_gradient_accumulation_steps,
                    "weight_decay": ankh_weight_decay,
                }
            experiment = f"{ssp}_best_{model}"
            if ssp == "ssp3":
                label_type = "label3"
            elif ssp == "ssp8":
                label_type = "label8"
            # Load data for the current model and task
            (
                train_embeddings,
                train_labels,
                casp12_embeddings,
                casp12_labels,
                casp13_embeddings,
                casp13_labels,
                casp14_embeddings,
                casp14_labels,
                ts115_embeddings,
                ts115_labels,
                cb513_embeddings,
                cb513_labels,
            ) = load_data()

            unique_tags = set()
            for label_dict in train_labels:
                unique_tags.update(label_dict[label_type])
            tag2id = {tag: id for id, tag in enumerate(unique_tags)}
            id2tag = {id: tag for tag, id in tag2id.items()}

            def align_predictions(predictions: np.ndarray, label_ids: np.ndarray):
                preds = np.argmax(predictions, axis=2)

                batch_size, seq_len = preds.shape

                out_label_list = [[] for _ in range(batch_size)]
                preds_list = [[] for _ in range(batch_size)]

                for i in range(batch_size):
                    for j in range(seq_len):
                        if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                            out_label_list[i].append(id2tag[label_ids[i][j]])
                            preds_list[i].append(id2tag[preds[i][j]])

                return preds_list, out_label_list

            def compute_metrics(p: EvalPrediction):
                preds_list, out_label_list = align_predictions(
                    p.predictions, p.label_ids
                )
                flat_preds_list = [label for sublist in preds_list for label in sublist]
                flat_out_label_list = [
                    label for sublist in out_label_list for label in sublist
                ]

                return {
                    "accuracy": accuracy_score(flat_out_label_list, flat_preds_list),
                    "precision": precision_score(
                        flat_out_label_list, flat_preds_list, average="macro"
                    ),
                    "recall": recall_score(
                        flat_out_label_list, flat_preds_list, average="macro"
                    ),
                    "f1": f1_score(
                        flat_out_label_list, flat_preds_list, average="macro"
                    ),
                }

            def encode_tags(label_dicts, key, tag2id):
                """Encodes tags in the list of label dictionaries for a specific key."""
                encoded_labels = []
                for label_dict in label_dicts:
                    labels = label_dict[key]
                    encoded_labels.append([tag2id[tag] for tag in labels])
                return encoded_labels

            train_labels_encodings = encode_tags(train_labels, label_type, tag2id)
            casp12_labels_encodings = encode_tags(casp12_labels, label_type, tag2id)
            casp13_labels_encodings = encode_tags(casp13_labels, label_type, tag2id)
            casp14_labels_encodings = encode_tags(casp14_labels, label_type, tag2id)
            ts115_labels_encodings = encode_tags(ts115_labels, label_type, tag2id)
            cb513_labels_encodings = encode_tags(cb513_labels, label_type, tag2id)

            train_disorders = [label_dict["disorder"] for label_dict in train_labels]
            train_labels_encodings = mask_disorder(
                train_labels_encodings, train_disorders
            )
            casp12_disorders = [label_dict["disorder"] for label_dict in casp12_labels]
            casp12_labels_encodings = mask_disorder(
                casp12_labels_encodings, casp12_disorders
            )
            casp13_disorders = [label_dict["disorder"] for label_dict in casp13_labels]
            casp13_labels_encodings = mask_disorder(
                casp13_labels_encodings, casp13_disorders
            )
            casp14_disorders = [label_dict["disorder"] for label_dict in casp14_labels]
            casp14_labels_encodings = mask_disorder(
                casp14_labels_encodings, casp14_disorders
            )
            ts115_disorders = [label_dict["disorder"] for label_dict in ts115_labels]
            ts115_labels_encodings = mask_disorder(
                ts115_labels_encodings, ts115_disorders
            )
            cb513_disorders = [label_dict["disorder"] for label_dict in cb513_labels]
            cb513_labels_encodings = mask_disorder(
                cb513_labels_encodings, cb513_disorders
            )

            indices = np.arange(len(train_embeddings))

            for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
                # Split data into training and validation for the current fold
                training_sequences = [train_embeddings[i] for i in train_idx]
                training_labels = [train_labels_encodings[i] for i in train_idx]
                validation_sequences = [train_embeddings[i] for i in val_idx]
                validation_labels = [train_labels_encodings[i] for i in val_idx]

                # Create training and validation datasets
                training_dataset, validation_dataset = create_datasets_ssp(
                    training_sequences,
                    training_labels,
                    validation_sequences,
                    validation_labels,
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
                    metric_for_best_model="eval_accuracy",
                    greater_is_better=True,
                    save_strategy="epoch",
                    report_to="wandb",
                )

                trainer = Trainer(
                    model_init=partial(
                        model_init,
                        num_tokens=len(unique_tags),
                        embed_dim=model_embed_dim,
                    ),
                    args=training_args,
                    train_dataset=training_dataset,
                    eval_dataset=validation_dataset,
                    compute_metrics=compute_metrics,
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
                            "ssp": ssp,
                            "fold": fold,
                            "metric": key,  # e.g., "accuracy", "precision", etc.
                            "value": eval_results[key],  # The value of the metric
                            "data_split": "cross-validation",
                        }
                    )

            final_trainer = Trainer(
                model_init=partial(
                    model_init,
                    num_tokens=len(unique_tags),
                    embed_dim=model_embed_dim,
                ),
                args=training_args,
                train_dataset=create_datasets_ssp(
                    train_embeddings, train_labels_encodings
                ),
                eval_dataset=create_datasets_ssp(
                    casp12_embeddings, casp12_labels_encodings
                ),
                compute_metrics=compute_metrics,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=5, early_stopping_threshold=0.05
                    )
                ],
            )

            print(f"Training {experiment} on the full training set...")

            # Train the model on the full training set
            final_trainer.train()

            # After cross-validation, evaluate on the test set
            casp12_dataset, casp13_dataset = create_datasets_ssp(
                casp12_embeddings,
                casp12_labels_encodings,
                casp13_embeddings,
                casp13_labels_encodings,
            )
            casp14_dataset, ts115_dataset = create_datasets_ssp(
                casp14_embeddings,
                casp14_labels_encodings,
                ts115_embeddings,
                ts115_labels_encodings,
            )
            cb513_dataset = SSPDataset(cb513_embeddings, cb513_labels_encodings)

            casp12_results = final_trainer.evaluate(casp12_dataset)
            casp13_results = final_trainer.evaluate(casp13_dataset)
            casp14_results = final_trainer.evaluate(casp14_dataset)
            ts115_results = final_trainer.evaluate(ts115_dataset)
            cb513_results = final_trainer.evaluate(cb513_dataset)

            # Store test set results
            # For test set results
            for dataset_name, dataset_results in [
                ("casp12", casp12_results),
                ("casp13", casp13_results),
                ("casp14", casp14_results),
                ("ts115", ts115_results),
                ("cb513", cb513_results),
            ]:
                for key in dataset_results.keys():
                    results.append(
                        {
                            "model": model,
                            "task": task,
                            "ssp": ssp,
                            "metric": key,  # e.g., "accuracy", "precision", etc.
                            "value": dataset_results[key],  # The value of the metric
                            "data_split": dataset_name,  # Name of the test dataset
                        }
                    )

    # Convert results to DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"./results/cross_validation_results_ssp.csv", index=False)


if __name__ == "__main__":
    main()
