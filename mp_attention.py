import os
import torch
import h5py
import logging
from transformers import T5TokenizerFast, T5EncoderModel, set_seed
from datasets import load_dataset
from tqdm import tqdm
import fire
import random
import numpy as np
import pandas as pd


def extract_all_attention_scores(model, tokenizer, sequences, device):
    # Get the total number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    attention_scores = {
        f"layer_{l}_head_{h}": [] for l in range(num_layers) for h in range(num_heads)
    }
    progress_bar = tqdm(sequences)
    with torch.no_grad():
        for sample in progress_bar:
            try:
                ids = tokenizer.batch_encode_plus(
                    [sample],
                    add_special_tokens=True,
                    padding=True,
                    is_split_into_words=True,
                    return_tensors="pt",
                )
                outputs = model(input_ids=ids["input_ids"].to(device))
                attentions = outputs.attentions
                for l in range(num_layers):
                    for h in range(num_heads):
                        specific_attention = attentions[l][0, h].detach().cpu().numpy()
                        attention_scores[f"layer_{l}_head_{h}"].append(
                            specific_attention
                        )
            except Exception as e:
                logging.error(f"Error processing sequence: {sample} - Error: {e}")
                continue
            progress_bar.close()
    return attention_scores


def save_specific_attention_with_index_as_id(filename, attentions, labels):
    with h5py.File(filename, "w") as f:
        for key, attention_list in attentions.items():
            for i, attention in enumerate(attention_list):
                ds_name = f"{key}_{i}"
                f.create_dataset(ds_name, data=attention)
                # Convert labels to fixed-length ASCII strings
                encoded_label = np.string_(labels[i])
                f[ds_name].attrs["label"] = encoded_label


def main(dataset_name="mp", seed=7):
    models = {"p2s": "./best_model_p2s_integration", "ankh": "ElnaggarLab/ankh-base"}

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Available device: {device}")

    # Load datasets from CSV files
    test_df = pd.read_csv("./datasets/mp/test.csv")
    logging.info("Dataset loaded")

    test_sequences, test_labels = (
        test_df["sequence"].tolist(),
        test_df["label"].tolist(),
    )

    def preprocess_dataset(sequences, labels, max_length=None):
        if max_length is None:
            max_length = len(max(sequences, key=lambda x: len(x)))
        return [list(seq[:max_length]) for seq in sequences], labels

    for model_short_name, model_path in models.items():
        logging.info(f"Processing with model: {model_short_name}")

        # Load model and tokenizer based on model_short_name
        if model_short_name == "p2s":
            model = T5EncoderModel.from_pretrained(model_path, output_attentions=True)
            tokenizer = T5TokenizerFast.from_pretrained(model_path)
        elif model_short_name == "ankh":
            model = T5EncoderModel.from_pretrained(model_path, output_attentions=True)
            tokenizer = T5TokenizerFast.from_pretrained(model_path)

        model.to(device).eval()

        test_sequences_processed, test_labels_processed = preprocess_dataset(
            test_sequences, test_labels
        )

        logging.info("Starting to process test sequences")
        test_attentions = extract_all_attention_scores(
            model, tokenizer, test_sequences_processed, device
        )
        logging.info("Test sequences processed")

        attentions_dir = f"./attentions/{model_short_name}_{dataset_name}"
        os.makedirs(attentions_dir, exist_ok=True)

        save_specific_attention_with_index_as_id(
            os.path.join(attentions_dir, "test_attention.h5py"),
            test_attentions,
            test_labels_processed,
        )

        logging.info(f"Completed processing for model: {model_short_name}")


if __name__ == "__main__":
    fire.Fire(main)
