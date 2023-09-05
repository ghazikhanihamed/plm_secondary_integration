import torch
import numpy as np
import random

seed = 7

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

import ankh

from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    AutoTokenizer,
    T5EncoderModel,
)
from datasets import load_dataset
from accelerate import Accelerator

from sklearn import metrics
from scipy import stats
from functools import partial
import pandas as pd
from tqdm.auto import tqdm

import wandb
import json

# Load Weights & Biases Configuration
with open("wandb_config.json") as f:
    data = json.load(f)

api_key = data["api_key"]

wandb_config = {
    "project": "plm_secondary_integration",
}

wandb.login(key=api_key)


def get_model_and_tokenizer(model_name):
    """Loads the model, puts it on the device and in eval mode, and retrieves the tokenizer."""
    model = T5EncoderModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


model, tokenizer = get_model_and_tokenizer("ghazikhanihamed/TooT-PLM-P2S")
model.eval()

dataset = load_dataset("proteinea/Solubility")

training_sequences, training_labels = (
    dataset["train"]["sequences"],
    dataset["train"]["labels"],
)
validation_sequences, validation_labels = (
    dataset["validation"]["sequences"],
    dataset["validation"]["labels"],
)
test_sequences, test_labels = dataset["test"]["sequences"], dataset["test"]["labels"]


def preprocess_dataset(sequences, labels, max_length=None):
    """
    Args:
        sequences: list, the list which contains the protein primary sequences.
        labels: list, the list which contains the dataset labels.
        max_length, Integer, the maximum sequence length,
        if there is a sequence that is larger than the specified sequence length will be post-truncated.
    """
    if max_length is None:
        max_length = len(max(training_sequences, key=lambda x: len(x)))
    splitted_sequences = [list(seq[:max_length]) for seq in sequences]
    return splitted_sequences, labels


def embed_dataset(model, sequences, shift_left=0, shift_right=-1):
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
            embedding = model(input_ids=ids["input_ids"])[0]
            embedding = embedding[0].detach().cpu().numpy()[shift_left:shift_right]
            inputs_embedding.append(embedding)
    return inputs_embedding


training_sequences, training_labels = preprocess_dataset(
    training_sequences, training_labels
)
validation_sequences, validation_labels = preprocess_dataset(
    validation_sequences, validation_labels
)
test_sequences, test_labels = preprocess_dataset(test_sequences, test_labels)

training_embeddings = embed_dataset(model, training_sequences)
validation_embeddings = embed_dataset(model, validation_sequences)
test_embeddings = embed_dataset(model, test_sequences)


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


training_dataset = SolubilityDataset(training_embeddings, training_labels)
validation_dataset = SolubilityDataset(validation_embeddings, validation_labels)
test_dataset = SolubilityDataset(test_embeddings, test_labels)


def model_init(embed_dim):
    hidden_dim = int(embed_dim / 2)
    num_hidden_layers = 1  # Number of hidden layers in ConvBert.
    nlayers = 1  # Number of ConvBert layers.
    nhead = 4
    dropout = 0.2
    conv_kernel_size = 7
    pooling = "max"  # available pooling methods ['avg', 'max']
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
    return downstream_model


def compute_metrics(p: EvalPrediction):
    preds = (torch.sigmoid(torch.tensor(p.predictions)).numpy() > 0.5).tolist()
    labels = p.label_ids.tolist()
    return {
        "accuracy": metrics.accuracy_score(labels, preds),
        "precision": metrics.precision_score(labels, preds),
        "recall": metrics.recall_score(labels, preds),
        "f1": metrics.f1_score(labels, preds),
    }


model_type = "ankh_base"
experiment = f"solubility_{model_type}"

training_args = TrainingArguments(
    output_dir=f"./results_{experiment}",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=1000,
    learning_rate=1e-03,
    weight_decay=0.0,
    logging_dir=f"./logs_{experiment}",
    logging_steps=200,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=4,
    fp16=False,
    fp16_opt_level="02",
    run_name=experiment,
    seed=seed,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    save_strategy="epoch",
)

model_embed_dim = 768  # Embedding dimension for ankh large.

trainer = Trainer(
    model_init=partial(model_init, embed_dim=model_embed_dim),
    args=training_args,
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(f"./best_model_{experiment}")

predictions, labels, metrics_output = trainer.predict(test_dataset)

# log metrics
print("Evaluation metrics: ", metrics_output)
