import torch
import os
import numpy as np
import random
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
)
import pandas as pd
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
seed = 7
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

model_type = "p2s_base"
experiment = f"ionchannel_classification_{model_type}"

# Load Weights & Biases Configuration
with open("wandb_config.json") as f:
    data = json.load(f)
api_key = data["api_key"]
wandb_config = {"project": "plm_secondary_integration"}
wandb.login(key=api_key)

# Load dataset
train_df = pd.read_csv("./dataset/ionchannels_membraneproteins_imbalanced_train.csv")
test_df = pd.read_csv("./dataset/ionchannels_membraneproteins_imbalanced_test.csv")

train_texts = train_df["sequence"].tolist()
train_labels = (
    train_df["label"].apply(lambda x: 1 if x == "ionchannels" else 0).tolist()
)
test_texts = test_df["sequence"].tolist()
test_labels = test_df["label"].apply(lambda x: 1 if x == "ionchannels" else 0).tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.1, stratify=train_labels, random_state=seed
)

# Compute sequence lengths
sequence_lengths = [len(seq) for seq in train_texts]
# Get the 99th percentile of sequence lengths
max_length_99 = int(np.percentile(sequence_lengths, 99))


# As your sequence lengths can be different from the previous dataset, we calculate the max length again
# max_length = len(max(train_texts, key=lambda x: len(x)))
max_length = max_length_99


# Load model and tokenizer
def get_model_and_tokenizer(model_name):
    model = T5EncoderModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


model, tokenizer = get_model_and_tokenizer("ghazikhanihamed/TooT-PLM-P2S")
model.eval()


def preprocess_dataset(sequences, labels, max_length=None):
    splitted_sequences = [list(seq[:max_length]) for seq in sequences]
    return splitted_sequences, labels


training_sequences, training_labels = preprocess_dataset(
    train_texts, train_labels, max_length
)
validation_sequences, validation_labels = preprocess_dataset(
    val_texts, val_labels, max_length
)
test_sequences, test_labels = preprocess_dataset(test_texts, test_labels, max_length)


# Embed dataset
def embed_dataset(model, sequences, dataset_name, shift_left=0, shift_right=-1):
    embed_dir = f"./embeddings/{experiment}"
    os.makedirs(embed_dir, exist_ok=True)
    embed_file = os.path.join(embed_dir, f"{dataset_name}_embeddings.npy")

    # Check if embeddings already exist
    if os.path.exists(embed_file):
        print(f"Loading {dataset_name} embeddings from disk...")
        return np.load(embed_file, allow_pickle=True)

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

    print(f"Saving {dataset_name} embeddings to disk...")
    np.save(embed_file, inputs_embedding)
    return inputs_embedding


training_embeddings = embed_dataset(model, training_sequences, "training")
validation_embeddings = embed_dataset(model, validation_sequences, "validation")
test_embeddings = embed_dataset(model, test_sequences, "test")


# Define custom dataset class
class IonChannelDataset(Dataset):  # Renamed for clarity
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


training_dataset = IonChannelDataset(training_embeddings, training_labels)
validation_dataset = IonChannelDataset(validation_embeddings, validation_labels)
test_dataset = IonChannelDataset(test_embeddings, test_labels)


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
        "mcc": metrics.matthews_corrcoef(labels, preds),
    }


training_args = TrainingArguments(
    output_dir=f"./results_{experiment}",
    num_train_epochs=10,
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
    metric_for_best_model="eval_mcc",
    greater_is_better=True,
    save_strategy="epoch",
    report_to="wandb",
)

model_embed_dim = 768  # Embedding dimension for ankh large.

# Initialize Trainer
trainer = Trainer(
    model_init=partial(model_init, embed_dim=model_embed_dim),
    args=training_args,
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the best model
trainer.save_model(f"./best_model_{experiment}")

# Make predictions and log metrics
predictions, labels, metrics_output = trainer.predict(test_dataset)
print("Evaluation metrics: ", metrics_output)
