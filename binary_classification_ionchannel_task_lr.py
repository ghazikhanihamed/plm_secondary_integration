import torch
import os
import numpy as np
import wandb
import json
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    T5EncoderModel,
    set_seed,
)
import pandas as pd
from sklearn.model_selection import train_test_split
import accelerate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


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
    wandb_config = {"project": "plm_secondary_integration"}
    wandb.login(key=api_key)
    wandb.init(project="plm_secondary_integration")


def setup_accelerate():
    # Setup Accelerate
    return accelerate.Accelerator(log_with=["wandb"])


def load_data():
    # Load dataset
    train_df = pd.read_csv(
        "./dataset/ionchannels_membraneproteins_imbalanced_train.csv"
    )
    test_df = pd.read_csv("./dataset/ionchannels_membraneproteins_imbalanced_test.csv")

    train_texts = train_df["sequence"].tolist()
    train_labels = (
        train_df["label"].apply(lambda x: 1 if x == "ionchannels" else 0).tolist()
    )
    test_texts = test_df["sequence"].tolist()
    test_labels = (
        test_df["label"].apply(lambda x: 1 if x == "ionchannels" else 0).tolist()
    )

    return train_texts, train_labels, test_texts, test_labels


def combine_and_split_data(train_texts, train_labels, test_texts, test_labels):
    # Combine train and test sequences
    combined_texts = train_texts + test_texts

    # Compute the max_length using the combined sequences
    sequence_lengths = [len(seq) for seq in combined_texts]
    max_length = int(np.percentile(sequence_lengths, 100))

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=0.1,
        stratify=train_labels,
        random_state=seed,
    )

    return train_texts, val_texts, train_labels, val_labels, max_length


def load_model_and_tokenizer(model_name, accelerator):
    # Load model and tokenizer
    model = T5EncoderModel.from_pretrained(model_name).to(accelerator.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return accelerator.prepare(model), tokenizer


def preprocess_dataset(sequences, labels, max_length=None):
    splitted_sequences = [list(seq[:max_length]) for seq in sequences]
    return splitted_sequences, labels


def embed_dataset(
    model,
    sequences,
    dataset_name,
    tokenizer,
    accelerator,
    experiment,
    shift_left=0,
    shift_right=-1,
):
    embed_dir = f"../embeddings/{experiment}"
    os.makedirs(embed_dir, exist_ok=True)
    embed_file = os.path.join(
        embed_dir, f"{dataset_name}_embeddings.pt"
    )  # Use .pt for PyTorch tensor

    # Check if embeddings already exist
    if os.path.exists(embed_file):
        accelerator.print(f"Loading {dataset_name} embeddings from disk...")
        tensor_data = torch.load(embed_file)
        if tensor_data.is_cuda:  # Ensure the tensor is on CPU
            tensor_data = tensor_data.cpu()
        return np.array(tensor_data.numpy())

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
            ids = {k: v.to(accelerator.device) for k, v in ids.items()}
            embedding = model(input_ids=ids["input_ids"])[0]
            embedding = (
                embedding[0].detach().cpu()[shift_left:shift_right].numpy().flatten()
            )
            inputs_embedding.append(embedding)

    accelerator.print(f"Saving {dataset_name} embeddings to disk...")
    torch.save(inputs_embedding, embed_file)  # Save the list of tensors
    return np.array(inputs_embedding)


def main():
    model_type = "ankh_base"
    experiment = f"ionchannel_classification_{model_type}"

    api_key = load_wandb_config()
    setup_wandb(api_key)
    accelerator = setup_accelerate()

    train_texts, train_labels, test_texts, test_labels = load_data()
    (
        train_texts,
        val_texts,
        train_labels,
        val_labels,
        max_length,
    ) = combine_and_split_data(train_texts, train_labels, test_texts, test_labels)

    model, tokenizer = load_model_and_tokenizer("ElnaggarLab/ankh-base", accelerator)

    training_sequences, training_labels = preprocess_dataset(
        train_texts, train_labels, max_length
    )
    validation_sequences, validation_labels = preprocess_dataset(
        val_texts, val_labels, max_length
    )
    test_sequences, test_labels = preprocess_dataset(
        test_texts, test_labels, max_length
    )

    training_embeddings = embed_dataset(
        model, training_sequences, "training", tokenizer, accelerator, experiment
    )
    validation_embeddings = embed_dataset(
        model, validation_sequences, "validation", tokenizer, accelerator, experiment
    )

    # Combine training and validation embeddings
    combined_embeddings = np.vstack((training_embeddings, validation_embeddings))

    # Combine training and validation labels
    combined_labels = training_labels + validation_labels

    test_embeddings = embed_dataset(
        model, test_sequences, "test", tokenizer, accelerator, experiment
    )

    # Initialize and train
    lr_model = LogisticRegression(random_state=1)
    lr_model.fit(combined_embeddings, combined_labels)

    # Predict
    test_preds = lr_model.predict(test_embeddings)

    # Calculate metrics for test set
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    test_mcc = matthews_corrcoef(test_labels, test_preds)

    # Print or log your metrics as needed
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Precision: {test_precision}")
    print(f"Test Recall: {test_recall}")
    print(f"Test F1-Score: {test_f1}")
    print(f"Test MCC: {test_mcc}")


if __name__ == "__main__":
    main()
