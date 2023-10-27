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
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

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
    wandb_config = {"project": "plm_secondary_integration_ssp3"}
    wandb.login(key=api_key)
    wandb.init(project="plm_secondary_integration_ssp3")


def setup_accelerate():
    # Setup Accelerate
    return accelerate.Accelerator(log_with=["wandb"])


def load_data():
    # Load dataset from Hugging Face
    training_dataset = load_dataset(
        "proteinea/SSP", data_files={"train": ["training_hhblits.csv"]}
    )
    casp12_dataset = load_dataset("proteinea/SSP", data_files={"test": ["CASP12.csv"]})
    casp14_dataset = load_dataset("proteinea/SSP", data_files={"test": ["CASP14.csv"]})
    ts115_dataset = load_dataset("proteinea/SSP", data_files={"test": ["TS115.csv"]})
    cb513_dataset = load_dataset("proteinea/SSP", data_files={"test": ["CB513.csv"]})

    input_column_name = "input"
    labels_column_name = (
        "dssp3"  # You can change it to "dssp8" if you want to work with 8 states.
    )
    disorder_column_name = "disorder"
    training_sequences, training_labels, training_disorder = (
        training_dataset["train"][input_column_name],
        training_dataset["train"][labels_column_name],
        training_dataset["train"][disorder_column_name],
    )

    casp12_sequences, casp12_labels, casp12_disorder = (
        casp12_dataset["test"][input_column_name],
        casp12_dataset["test"][labels_column_name],
        casp12_dataset["test"][disorder_column_name],
    )

    casp14_sequences, casp14_labels, casp14_disorder = (
        casp14_dataset["test"][input_column_name],
        casp14_dataset["test"][labels_column_name],
        casp14_dataset["test"][disorder_column_name],
    )

    ts115_sequences, ts115_labels, ts115_disorder = (
        ts115_dataset["test"][input_column_name],
        ts115_dataset["test"][labels_column_name],
        ts115_dataset["test"][disorder_column_name],
    )

    cb513_sequences, cb513_labels, cb513_disorder = (
        cb513_dataset["test"][input_column_name],
        cb513_dataset["test"][labels_column_name],
        cb513_dataset["test"][disorder_column_name],
    )

    return (
        training_sequences,
        training_labels,
        training_disorder,
        casp12_sequences,
        casp12_labels,
        casp12_disorder,
        casp14_sequences,
        casp14_labels,
        casp14_disorder,
        ts115_sequences,
        ts115_labels,
        ts115_disorder,
        cb513_sequences,
        cb513_labels,
        cb513_disorder,
    )


def load_model_and_tokenizer(model_name):
    # Load model and tokenizer
    device = get_device()
    model = T5EncoderModel.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def preprocess_dataset(sequences, labels, disorder, max_length=None):
    sequence_length = [len(seq) for seq in sequences]
    max_length = int(np.percentile(sequence_length, 99))
    print("Max length: ", max_length)

    sequences = ["".join(seq.split()) for seq in sequences]

    seqs = [list(seq)[:max_length] for seq in sequences]

    labels = ["".join(label.split()) for label in labels]
    labels = [list(label)[:max_length] for label in labels]

    disorder = [" ".join(disorder.split()) for disorder in disorder]
    disorder = [disorder.split()[:max_length] for disorder in disorder]

    assert len(seqs) == len(labels) == len(disorder)
    return seqs, labels, disorder


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
    training_labels,
    training_disorder,
    casp12_sequences,
    casp12_labels,
    casp12_disorder,
    casp14_sequences,
    casp14_labels,
    casp14_disorder,
    ts115_sequences,
    ts115_labels,
    ts115_disorder,
    cb513_sequences,
    cb513_labels,
    cb513_disorder,
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
    casp14_dataset = SSPDataset(casp14_sequences, casp14_labels)
    ts115_dataset = SSPDataset(ts115_sequences, ts115_labels)
    cb513_dataset = SSPDataset(cb513_sequences, cb513_labels)

    return (
        training_dataset,
        casp12_dataset,
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


def main():
    model_type = "p2s"
    experiment = f"ssp3_{model_type}"

    api_key = load_wandb_config()
    setup_wandb(api_key)
    # accelerator = setup_accelerate()

    (
        training_sequences,
        training_labels,
        training_disorder,
        casp12_sequences,
        casp12_labels,
        casp12_disorder,
        casp14_sequences,
        casp14_labels,
        casp14_disorder,
        ts115_sequences,
        ts115_labels,
        ts115_disorder,
        cb513_sequences,
        cb513_labels,
        cb513_disorder,
    ) = load_data()

    model, tokenizer = load_model_and_tokenizer("ghazikhanihamed/TooT-PLM-P2S")

    training_sequences, training_labels, training_disorder = preprocess_dataset(
        training_sequences, training_labels, training_disorder
    )
    casp12_sequences, casp12_labels, casp12_disorder = preprocess_dataset(
        casp12_sequences, casp12_labels, casp12_disorder
    )

    casp14_sequences, casp14_labels, casp14_disorder = preprocess_dataset(
        casp14_sequences, casp14_labels, casp14_disorder
    )
    ts115_sequences, ts115_labels, ts115_disorder = preprocess_dataset(
        ts115_sequences, ts115_labels, ts115_disorder
    )
    cb513_sequences, cb513_labels, cb513_disorder = preprocess_dataset(
        cb513_sequences, cb513_labels, cb513_disorder
    )

    training_embeddings = embed_dataset(
        model, training_sequences, "training", tokenizer, experiment
    )
    casp12_embeddings = embed_dataset(
        model, casp12_sequences, "casp12", tokenizer, experiment
    )
    casp14_embeddings = embed_dataset(
        model, casp14_sequences, "casp14", tokenizer, experiment
    )
    ts115_embeddings = embed_dataset(
        model, ts115_sequences, "ts115", tokenizer, experiment
    )
    cb513_embeddings = embed_dataset(
        model, cb513_sequences, "cb513", tokenizer, experiment
    )

    unique_tags = set(tag for doc in training_labels for tag in doc)
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
        return {
            "accuracy": accuracy_score(out_label_list, preds_list),
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

    def encode_tags(labels, tag2id):
        labels = [[tag2id[tag] for tag in doc] for doc in labels]
        return labels

    train_labels_encodings = encode_tags(training_labels, tag2id)
    casp12_labels_encodings = encode_tags(casp12_labels, tag2id)
    casp14_labels_encodings = encode_tags(casp14_labels, tag2id)
    ts115_labels_encodings = encode_tags(ts115_labels, tag2id)
    cb513_labels_encodings = encode_tags(cb513_labels, tag2id)

    train_labels_encodings = mask_disorder(train_labels_encodings, training_disorder)
    casp12_labels_encodings = mask_disorder(casp12_labels_encodings, casp12_disorder)
    casp14_labels_encodings = mask_disorder(casp14_labels_encodings, casp14_disorder)
    ts115_labels_encodings = mask_disorder(ts115_labels_encodings, ts115_disorder)
    cb513_labels_encodings = mask_disorder(cb513_labels_encodings, cb513_disorder)

    (
        training_dataset,
        casp12_dataset,
        casp14_dataset,
        ts115_dataset,
        cb513_dataset,
    ) = create_datasets(
        training_sequences,
        training_labels,
        training_disorder,
        casp12_sequences,
        casp12_labels,
        casp12_disorder,
        casp14_sequences,
        casp14_labels,
        casp14_disorder,
        ts115_sequences,
        ts115_labels,
        ts115_disorder,
        cb513_sequences,
        cb513_labels,
        cb513_disorder,
    )

    model_embed_dim = 768

    training_args = TrainingArguments(
        output_dir=f"./results_{experiment}",
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=1000,
        learning_rate=1e-03,
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

    # Train the model
    trainer.train()

    # Save the best model
    trainer.save_model(f"./best_model_{experiment}")

    # Make predictions and log metrics
    predictions, labels, metrics_output = trainer.predict(casp12_dataset)
    print("Evaluation metrics on casp12: ", metrics_output)

    predictions, labels, metrics_output = trainer.predict(casp14_dataset)
    print("Evaluation metrics on casp14: ", metrics_output)

    predictions, labels, metrics_output = trainer.predict(ts115_dataset)
    print("Evaluation metrics on ts115: ", metrics_output)

    predictions, labels, metrics_output = trainer.predict(cb513_dataset)
    print("Evaluation metrics on cb513: ", metrics_output)


if __name__ == "__main__":
    main()
