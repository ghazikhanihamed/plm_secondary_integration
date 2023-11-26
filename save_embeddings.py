import torch
import pandas as pd
import os
import json
from transformers import T5TokenizerFast, T5EncoderModel, set_seed
from tqdm.auto import tqdm
import wandb
import logging

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

set_seed(42)
logging.basicConfig(filename="error_log.log", level=logging.ERROR)


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
    wandb_config = {"project": "save_embeddings"}
    wandb.login(key=api_key)
    wandb.init(project="save_embeddings")


# Function to load model and tokenizer
def load_model_and_tokenizer(model_name, local_path=None):
    device = get_device()
    if local_path:
        model = T5EncoderModel.from_pretrained(local_path).to(device).eval()
    else:
        model = T5EncoderModel.from_pretrained(model_name).to(device).eval()
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    return model, tokenizer


# Function to preprocess dataset
def preprocess_dataset(sequences, max_length=None):
    if max_length is None:
        max_length = len(max(sequences, key=lambda x: len(x)))
    splitted_sequences = [list(seq[:max_length]) for seq in sequences]
    return splitted_sequences


# Function to embed dataset
def embed_dataset(model, sequences, tokenizer, shift_left=0, shift_right=-1):
    device = get_device()
    inputs_embedding = []
    try:
        with torch.no_grad():
            for sample in tqdm(sequences):
                # Check if the sample is a list of tokenized words
                if isinstance(sample, list):
                    # If sample is a list of tokenized words
                    ids = tokenizer.batch_encode_plus(
                        sample,  # Directly pass the list of tokenized words
                        add_special_tokens=True,
                        padding=True,
                        is_split_into_words=True,
                        return_tensors="pt",
                    )
                else:
                    # If sample is a single string sequence
                    ids = tokenizer.batch_encode_plus(
                        [sample],  # Wrap the string in a list
                        add_special_tokens=True,
                        padding=True,
                        is_split_into_words=False,
                        return_tensors="pt",
                    )
                embedding = model(input_ids=ids["input_ids"].to(device))[0]
                embedding = embedding[0].detach().cpu().numpy()[shift_left:shift_right]
                inputs_embedding.append(embedding)
    except Exception as e:
        logging.error(f"Error embedding sequence with length {len(sample)}")
        logging.error(f"Exception: {e}")
        logging.error("Traceback:", exc_info=True)
        wandb.log({"error": str(e)})
    return inputs_embedding


# Function to save embeddings and additional data
def save_embeddings(embeddings, additional_data, filename):
    os.makedirs("./embeddings", exist_ok=True)
    filepath = os.path.join("./embeddings", filename)
    torch.save({"embeddings": embeddings, "additional_data": additional_data}, filepath)


# Function to process a given dataset
def process_and_save_dataset(dataset_path, sequence_col, label_cols, models):
    for file in os.listdir(dataset_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(dataset_path, file))
            sequences = df[sequence_col].tolist()

            try:
                # Process data for each model
                for model_name, model_details in models.items():
                    print(
                        f"Processing file {dataset_path}, dataset {file} with model {model_name}"
                    )
                    model, tokenizer = model_details
                    # splitted_sequences = preprocess_dataset(sequences)
                    embeddings = embed_dataset(model, sequences, tokenizer)

                    # Save embeddings with each label column
                    for label_col in label_cols:
                        labels = df[label_col].values
                        save_embeddings(
                            embeddings,
                            labels,
                            f"{file}_{model_name}_{label_col}_embeddings.pt",
                        )
            except Exception as e:
                logging.error(f"Error processing file {dataset_path}, dataset {file}")
                logging.error(f"Exception: {e}")
                logging.error("Traceback:", exc_info=True)
                wandb.log({"error": str(e)})
    print(f"Finished processing dataset at {dataset_path}")


# Main function to process and save embeddings for multiple datasets
def main():
    api_key = load_wandb_config()
    setup_wandb(api_key)

    # Load models and tokenizers
    models = {
        "p2s": load_model_and_tokenizer(
            "ghazikhanihamed/TooT-PLM-P2S", "./best_model_p2s_integration"
        ),
        "ankh": load_model_and_tokenizer("ElnaggarLab/ankh-base"),
    }

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
        process_and_save_dataset(
            dataset_path, details["sequence_col"], details["label_cols"], models
        )


if __name__ == "__main__":
    main()
