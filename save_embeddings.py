import torch
import pandas as pd
import os
import json
from transformers import AutoTokenizer, T5EncoderModel, AutoModel
from tqdm.auto import tqdm
from accelerate import Accelerator
import logging
import wandb
from accelerate.logging import get_logger

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)

logger = get_logger(__name__)

# Load Weights & Biases Configuration
with open("wandb_config.json") as f:
    data = json.load(f)

api_key = data["api_key"]

wandb.login(key=api_key)


# Function to determine available device
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to load model and tokenizer
def load_model_and_tokenizer(model_name, accelerator, local_path=None):
    if local_path:
        model = T5EncoderModel.from_pretrained(local_path).to(accelerator.device).eval()
    else:
        model = AutoModel.from_pretrained(model_name).to(accelerator.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


# Function to preprocess dataset
def preprocess_dataset(sequences, tokenizer, max_length=None):
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    tokenized_sequences = tokenizer.batch_encode_plus(
        sequences,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return tokenized_sequences


# Function to embed dataset
def embed_dataset(model, tokenized_sequences, accelerator):
    with torch.no_grad():
        outputs = model(
            **{k: v.to(accelerator.device) for k, v in tokenized_sequences.items()}
        )
        embeddings = outputs.last_hidden_state.detach()
    return accelerator.gather(embeddings)


# Function to save embeddings and additional data
def save_embeddings(embeddings, additional_data, filename):
    os.makedirs("./embeddings", exist_ok=True)
    filepath = os.path.join("./embeddings", filename)
    torch.save({"embeddings": embeddings, "additional_data": additional_data}, filepath)


# Function to process a given dataset
def process_and_save_dataset(dataset_path, sequence_col, label_cols, models):
    logging.info(f"Processing dataset at {dataset_path}")
    for file in tqdm(
        os.listdir(dataset_path), desc=f"Processing files in {dataset_path}"
    ):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(dataset_path, file))
            sequences = df[sequence_col].tolist()

            # Process data for each model
            for model_name, model_details in models.items():
                model, tokenizer = model_details
                tokenized = preprocess_dataset(sequences, tokenizer)
                embeddings = embed_dataset(model, tokenized)

                # Save embeddings with each label column
                for label_col in label_cols:
                    labels = df[label_col].values
                    save_embeddings(
                        embeddings,
                        labels,
                        f"{file}_{model_name}_{label_col}_embeddings.pt",
                    )
    logging.info(f"Finished processing dataset at {dataset_path}")


# Main function to process and save embeddings for multiple datasets
def main():
    # Initialize Accelerator
    accelerator = Accelerator(log_with="wandb")

    accelerator.init_trackers(
        "save_embeddings",
        init_kwargs={
            "wandb": {
                "notes": "Embeddings for all datasets",
                "tags": ["embeddings"],
            }
        },
    )
    # Load models and tokenizers
    models = {
        "p2s": load_model_and_tokenizer(
            "ghazikhanihamed/TooT-PLM-P2S", accelerator, "./best_model_p2s_integration"
        ),
        "ankh": load_model_and_tokenizer("ElnaggarLab/ankh-base", accelerator),
    }

    # Initialize models with Accelerator and update the models dictionary
    for model_name in models:
        model, tokenizer = models[model_name]
        prepared_model = accelerator.prepare(model)
        models[model_name] = (
            prepared_model,
            tokenizer,
        )  # Update the model in the dictionary

    # Define datasets and their respective columns
    datasets = {
        "./datasets/solubility": {
            "sequence_col": "sequences",
            "label_cols": ["labels"],
        },
        "./datasets/localization": {"sequence_col": "input", "label_cols": ["loc"]},
        "./datasets/ssp": {
            "sequence_col": "input",
            "label_cols": ["dssp3", "dssp8", "disorder"],
        },
        "./datasets/fluorescence": {
            "sequence_col": "primary",
            "label_cols": ["log_fluorescence"],
        },
        "./datasets/ionchannels": {
            "sequence_col": "sequence",
            "label_cols": ["label", "id"],
        },
        "./datasets/mp": {"sequence_col": "sequence", "label_cols": ["label"]},
        "./datasets/transporters": {
            "sequence_col": "sequence",
            "label_cols": ["label"],
        },
    }

    # Process each dataset
    for dataset_path, details in datasets.items():
        logging.info(f"Processing dataset at {dataset_path}")
        process_and_save_dataset(
            dataset_path, details["sequence_col"], details["label_cols"], models
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
