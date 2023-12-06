import torch
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import ankh
import wandb
import json
from functools import partial
from sklearn import metrics
from torch.utils.data import Dataset, random_split
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
    EarlyStoppingCallback,
)
from load_embeddings import load_ssp_embeddings_and_labels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Set seed for reproducibility
seed = 7
set_seed(seed)


def load_wandb_config():
    # Load Weights & Biases Configuration
    with open("wandb_config.json") as f:
        data = json.load(f)
    return data["api_key"]


def setup_wandb(api_key):
    # Setup Weights & Biases
    wandb.login(key=api_key)


def load_data():
    return load_ssp_embeddings_and_labels("./embeddings/p2s_SSP")


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


def optuna_hp_space(trial):
    return {
        # Learning rate: For fine-tuning, a smaller range focused on smaller values is preferred.
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        # Weight decay: This range is reasonable for fine-tuning, but still not too high.
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1, step=0.01),
        # Warmup ratio: As before, a step is introduced to not have a very granular search in this space.
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.5, step=0.05),
        # Gradient accumulation steps: This is a very important hyperparameter to tune. It is used to simulate
        "gradient_accumulation_steps": trial.suggest_int(
            "gradient_accumulation_steps", 1, 16
        ),
    }


def compute_objective(metrics):
    # This function will compute the objective for optimization during hyperparameter tuning.
    return metrics["eval_accuracy"]


def main():
    model_type = "p2s"
    experiment = f"ssp3_hyperparam_{model_type}"

    api_key = load_wandb_config()
    setup_wandb(api_key)

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
        unique_tags.update(label_dict["label3"])
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
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        flat_preds_list = [label for sublist in preds_list for label in sublist]
        flat_out_label_list = [label for sublist in out_label_list for label in sublist]

        return {
            "accuracy": accuracy_score(flat_out_label_list, flat_preds_list),
            "precision": precision_score(
                flat_out_label_list, flat_preds_list, average="macro"
            ),
            "recall": recall_score(
                flat_out_label_list, flat_preds_list, average="macro"
            ),
            "f1": f1_score(flat_out_label_list, flat_preds_list, average="macro"),
        }

    def encode_tags(label_dicts, key, tag2id):
        """Encodes tags in the list of label dictionaries for a specific key."""
        encoded_labels = []
        for label_dict in label_dicts:
            labels = label_dict[key]
            encoded_labels.append([tag2id[tag] for tag in labels])
        return encoded_labels

    train_labels_encodings = encode_tags(train_labels, "label3", tag2id)
    casp12_labels_encodings = encode_tags(casp12_labels, "label3", tag2id)
    casp13_labels_encodings = encode_tags(casp13_labels, "label3", tag2id)
    casp14_labels_encodings = encode_tags(casp14_labels, "label3", tag2id)
    ts115_labels_encodings = encode_tags(ts115_labels, "label3", tag2id)
    cb513_labels_encodings = encode_tags(cb513_labels, "label3", tag2id)

    train_labels_encodings = mask_disorder(
        train_labels_encodings, train_labels["disorder"]
    )
    casp12_labels_encodings = mask_disorder(
        casp12_labels_encodings, casp12_labels["disorder"]
    )
    casp13_labels_encodings = mask_disorder(
        casp13_labels_encodings, casp13_labels["disorder"]
    )
    casp14_labels_encodings = mask_disorder(
        casp14_labels_encodings, casp14_labels["disorder"]
    )
    ts115_labels_encodings = mask_disorder(
        ts115_labels_encodings, ts115_labels["disorder"]
    )
    cb513_labels_encodings = mask_disorder(
        cb513_labels_encodings, cb513_labels["disorder"]
    )

    (
        training_dataset,
        casp12_dataset,
        casp13_dataset,
        casp14_dataset,
        ts115_dataset,
        cb513_dataset,
    ) = create_datasets(
        train_embeddings,
        train_labels_encodings,
        casp12_embeddings,
        casp12_labels_encodings,
        casp13_embeddings,
        casp13_labels_encodings,
        casp14_embeddings,
        casp14_labels_encodings,
        ts115_embeddings,
        ts115_labels_encodings,
        cb513_embeddings,
        cb513_labels_encodings,
    )

    model_embed_dim = 768

    training_args = TrainingArguments(
        output_dir=f"./results_{experiment}",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=1000,
        learning_rate=1e-03,
        logging_dir=f"./logs_{experiment}",
        logging_steps=200,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=16,
        fp16=False,
        fp16_opt_level="O2",
        run_name=experiment,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_strategy="epoch",
        report_to="wandb",
    )

    # Initialize Trainer
    trainer = Trainer(
        model_init=partial(
            model_init, num_tokens=len(unique_tags), embed_dim=model_embed_dim
        ),
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=casp12_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3, early_stopping_threshold=0.05
            )
        ],
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

    # Make predictions and log metrics
    predictions, labels, metrics_output = trainer.predict(casp14_dataset)
    print("Evaluation metrics: ", metrics_output)


if __name__ == "__main__":
    main()
