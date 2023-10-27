import torch
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import ankh
import wandb
import json
from functools import partial
from tqdm.auto import tqdm
from sklearn import metrics
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    AutoTokenizer,
    T5EncoderModel,
    set_seed,
    EarlyStoppingCallback,
)
import pandas as pd
from sklearn.model_selection import train_test_split
import accelerate
from datasets import load_dataset

# Set seed for reproducibility
seed = 7
set_seed(seed)


# Function to determine available device
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_wandb_config():
    # Load Weights & Biases Configuration
    with open("wandb_config.json") as f:
        data = json.load(f)
    return data["api_key"]


def setup_wandb(api_key):
    # Setup Weights & Biases
    wandb_config = {"project": "plm_secondary_integration_solubility"}
    wandb.login(key=api_key)
    wandb.init(project="plm_secondary_integration_solubility")


def setup_accelerate():
    # Setup Accelerate
    return accelerate.Accelerator(log_with=["wandb"])


def load_data():
    # Load dataset from Hugging Face
    dataset = load_dataset("proteinea/solubility")

    # Extract training, validation, and test sets
    training_sequences = dataset["train"]["sequences"]
    training_labels = dataset["train"]["labels"]

    validation_sequences = dataset["validation"]["sequences"]
    validation_labels = dataset["validation"]["labels"]

    test_sequences = dataset["test"]["sequences"]
    test_labels = dataset["test"]["labels"]

    return (
        training_sequences,
        training_labels,
        validation_sequences,
        validation_labels,
        test_sequences,
        test_labels,
    )


def load_model_and_tokenizer(model_name):
    # Load model and tokenizer
    device = get_device()
    model = T5EncoderModel.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def preprocess_dataset(training_sequences, sequences, labels, max_length=None):
    if max_length is None:
        max_length = len(max(training_sequences, key=lambda x: len(x)))
    splitted_sequences = [list(seq[:max_length]) for seq in sequences]
    return splitted_sequences, labels


def embed_dataset(
    model,
    sequences,
    dataset_name,
    tokenizer,
    experiment,
    shift_left=0,
    shift_right=-1,
):
    device = get_device()
    embed_dir = f"./embeddings/{experiment}"
    os.makedirs(embed_dir, exist_ok=True)
    embed_file = os.path.join(
        embed_dir, f"{dataset_name}_embeddings.pt"
    )  # Use .pt for PyTorch tensor

    # Check if embeddings already exist
    if os.path.exists(embed_file):
        print(f"Loading {dataset_name} embeddings from disk...")
        return torch.load(embed_file)  # Load using torch.load

    inputs_embedding = []
    with torch.no_grad():
        for sample in tqdm(sequences):
            ids = tokenizer.batch_encode_plus(
                [sample],
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
            ids = {k: v.to(device) for k, v in ids.items()}
            embedding = model(input_ids=ids["input_ids"])[0]
            embedding = embedding[0].detach().cpu()[shift_left:shift_right]
            inputs_embedding.append(embedding)

    print(f"Saving {dataset_name} embeddings to disk...")
    torch.save(inputs_embedding, embed_file)  # Save the list of tensors
    return inputs_embedding


def create_datasets(
    training_sequences,
    validation_sequences,
    test_sequences,
    training_labels,
    validation_labels,
    test_labels,
):
    class SolubilityDataset(Dataset):
        def __init__(self, sequences, labels):
            self.sequences = sequences
            self.labels = labels

        def __getitem__(self, idx):
            embedding = self.sequences[idx]
            label = self.labels[idx]
            return {
                "embed": torch.tensor(embedding),
                "labels": torch.tensor(label, dtype=torch.float32).unsqueeze(-1),
            }

        def __len__(self):
            return len(self.sequences)

    training_dataset = SolubilityDataset(training_sequences, training_labels)
    validation_dataset = SolubilityDataset(validation_sequences, validation_labels)
    test_dataset = SolubilityDataset(test_sequences, test_labels)

    return training_dataset, validation_dataset, test_dataset


def model_init(embed_dim):
    checkpoint_path = "./results_solubility_prediction_p2s/checkpoint-11714/pytorch_model.bin"
    state_dict = torch.load(checkpoint_path)
    hidden_dim = int(embed_dim / 2)
    num_hidden_layers = 1
    nlayers = 1
    nhead = 4
    dropout = 0.2
    conv_kernel_size = 7
    pooling = "max"
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
    downstream_model.load_state_dict(state_dict)
    return downstream_model


def compute_metrics(p: EvalPrediction):
    preds = (torch.sigmoid(torch.tensor(p.predictions)).numpy() > 0.5).tolist()
    labels = p.label_ids.tolist()
    return {
        "accuracy": metrics.accuracy_score(labels, preds),
        "precision": metrics.precision_score(labels, preds),
        "recall": metrics.recall_score(labels, preds),
        "f1": metrics.f1_score(labels, preds),
        "mcc": metrics.matthews_corrcoef(labels, preds),
    }


def main():
    model_type = "p2s"
    experiment = f"solubility_prediction_{model_type}"

    api_key = load_wandb_config()
    setup_wandb(api_key)
    # accelerator = setup_accelerate()

    (
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        test_texts,
        test_labels,
    ) = load_data()

    model, tokenizer = load_model_and_tokenizer("ghazikhanihamed/TooT-PLM-P2S")

    training_sequences, training_labels = preprocess_dataset(
        train_texts, train_texts, train_labels
    )
    validation_sequences, validation_labels = preprocess_dataset(
        train_texts, val_texts, val_labels
    )
    test_sequences, test_labels = preprocess_dataset(
        train_texts, test_texts, test_labels
    )

    training_embeddings = embed_dataset(
        model, training_sequences, "training", tokenizer, experiment
    )
    validation_embeddings = embed_dataset(
        model, validation_sequences, "validation", tokenizer, experiment
    )
    test_embeddings = embed_dataset(
        model, test_sequences, "test", tokenizer, experiment
    )

    training_dataset, validation_dataset, test_dataset = create_datasets(
        training_embeddings,
        validation_embeddings,
        test_embeddings,
        training_labels,
        validation_labels,
        test_labels,
    )

    model_embed_dim = 768

    training_args = TrainingArguments(
        output_dir=f"./results_{experiment}",
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=1000,
        learning_rate=1e-04,
        weight_decay=1e-05,
        logging_dir=f"./logs_{experiment}",
        logging_steps=200,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=16,
        fp16=False,
        fp16_opt_level="02",
        run_name=experiment,
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_strategy="epoch",
        report_to="wandb",
        # hub_token="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",
        # hub_model_id="ghazikhanihamed/TooT-PLM-P2S_ionchannels-membrane",
    )

    # Initialize Trainer
    trainer = Trainer(
        model_init=partial(model_init, embed_dim=model_embed_dim),
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

    # Train the model
    trainer.train()

    # Save the best model
    trainer.save_model(f"./best_model_{experiment}")

    # Make predictions and log metrics
    predictions, labels, metrics_output = trainer.predict(test_dataset)
    print("Evaluation metrics: ", metrics_output)


if __name__ == "__main__":
    main()
