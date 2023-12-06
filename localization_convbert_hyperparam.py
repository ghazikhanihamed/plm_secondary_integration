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
from load_embeddings import load_embeddings_and_labels
import numpy as np
from sklearn.preprocessing import LabelEncoder

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
    return load_embeddings_and_labels("./embeddings/p2s_localization", "localization")


def create_datasets(
    training_sequences,
    training_labels,
    test_sequences,
    test_labels,
    label_encoder=None,
):
    class LocalizationDataset(Dataset):
        def __init__(self, sequences, labels):
            self.sequences = sequences
            self.labels = labels

        def __getitem__(self, idx):
            embedding = self.sequences[idx]
            label = self.labels[idx]
            # Convert string label to integer
            label_int = label_encoder.transform([label])[0]
            return {
                "embed": torch.tensor(embedding),
                "labels": torch.tensor(label_int),
            }

        def __len__(self):
            return len(self.sequences)

        # Create the full training dataset

    full_training_dataset = LocalizationDataset(training_sequences, training_labels)

    # Calculate the size of training and validation sets
    total_train_samples = len(full_training_dataset)
    val_size = int(np.floor(0.1 * total_train_samples))
    train_size = total_train_samples - val_size

    # Randomly split the dataset into training and validation datasets
    training_dataset, validation_dataset = random_split(
        full_training_dataset, [train_size, val_size]
    )

    test_dataset = LocalizationDataset(test_sequences, test_labels)

    return training_dataset, validation_dataset, test_dataset


def model_init(num_tokens, embed_dim, class_weights=None):
    hidden_dim = int(embed_dim / 2)
    num_hidden_layers = 1
    nlayers = 1
    nhead = 4
    dropout = 0.2
    conv_kernel_size = 7
    downstream_model = ankh.ConvBertForMultiClassClassification(
        input_dim=embed_dim,
        nhead=nhead,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        num_layers=nlayers,
        kernel_size=conv_kernel_size,
        dropout=dropout,
    )
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32)
    )
    downstream_model.set_loss_fn(loss_fn)

    return downstream_model


def compute_metrics(p: EvalPrediction):
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
    return metrics["eval_f1"]


def compute_class_weights(labels):
    """
    Compute class weights based on the inverse frequency of each class.

    Args:
    labels (list): A list of integer-encoded class labels.

    Returns:
    numpy.ndarray: An array of class weights.
    """
    # Count the frequency of each class
    class_counts = np.bincount(labels)

    # Compute class weights as inverse of class frequencies
    class_weights = 1.0 / class_counts

    # Normalize weights so that the smallest weight is 1
    class_weights = class_weights / class_weights.min()

    return class_weights


def main():
    model_type = "p2s"
    experiment = f"localization_hyperparam_{model_type}"

    api_key = load_wandb_config()
    setup_wandb(api_key)

    (
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
    ) = load_data()

    # a list of all possible labels in your dataset
    all_labels = list(set(train_labels + test_labels))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    train_labels_encoded = label_encoder.transform(train_labels)
    class_weights = compute_class_weights(train_labels_encoded)

    training_dataset, validation_dataset, test_dataset = create_datasets(
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
        label_encoder,
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
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_strategy="epoch",
        report_to="wandb",
    )

    # Initialize Trainer
    trainer = Trainer(
        model_init=partial(
            model_init,
            len(np.unique(train_labels_encoded)),
            model_embed_dim,
            class_weights,
        ),
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
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
    predictions, labels, metrics_output = trainer.predict(test_dataset)
    print("Evaluation metrics: ", metrics_output)


if __name__ == "__main__":
    main()
