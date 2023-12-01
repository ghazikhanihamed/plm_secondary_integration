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


def main(
    model_name="./best_model_p2s_integration",
    model_short_name="p2s",
    dataset_name="ionchannels",
    seed=7,
):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Uncomment the following line to log to a file
    # logging.basicConfig(filename='process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting script")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Available device: {device}")

    model = T5EncoderModel.from_pretrained(model_name).to(device).eval()
    tokenizer = T5TokenizerFast.from_pretrained(model_name)

    def get_num_params(model):
        return sum(p.numel() for p in model.parameters())

    model.eval()
    model.to(device=device)
    logging.info(f"Number of parameters: {get_num_params(model)}")

    # Load datasets from CSV files
    train_df = pd.read_csv("./datasets/ionchannels/train.csv")
    test_df = pd.read_csv("./datasets/ionchannels/test.csv")
    logging.info("Dataset loaded")

    training_sequences, training_labels = (
        train_df["sequence"].tolist(),
        train_df["label"].tolist(),
    )
    test_sequences, test_labels = (
        test_df["sequence"].tolist(),
        test_df["label"].tolist(),
    )

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
        progress_bar = tqdm(enumerate(sequences), total=len(sequences))
        with torch.no_grad():
            for i, sample in progress_bar:
                try:
                    ids = tokenizer.batch_encode_plus(
                        [sample],
                        add_special_tokens=True,
                        padding=True,
                        is_split_into_words=True,
                        return_tensors="pt",
                    )
                    embedding = model(input_ids=ids["input_ids"].to(device))[0]
                    embedding = (
                        embedding[0].detach().cpu().numpy()[shift_left:shift_right]
                    )
                    inputs_embedding.append(embedding)
                except Exception as e:
                    logging.error(
                        f"Error processing sequence at index {i}: {sample} - Error: {e}"
                    )
                    continue  # Continue with the next sample
                finally:
                    progress_bar.set_description(f"Processed: {len(inputs_embedding)}")
        return inputs_embedding

    training_sequences, training_labels = preprocess_dataset(
        training_sequences, training_labels
    )
    assert len(training_sequences) == len(
        training_labels
    ), "Mismatch in number of training sequences and labels"

    test_sequences, test_labels = preprocess_dataset(test_sequences, test_labels)
    assert len(test_sequences) == len(
        test_labels
    ), "Mismatch in number of test sequences and labels"

    # Right before calling embed_dataset
    logging.info(f"Number of training sequences: {len(training_sequences)}")
    logging.info(f"Number of training labels: {len(training_labels)}")

    logging.info("Starting to process training sequences")
    training_embeddings = embed_dataset(model, training_sequences)
    logging.info("Training sequences processed")
    logging.info(f"Number of generated embeddings: {len(training_embeddings)}")
    logging.info("Training sequences processed")
    assert len(training_embeddings) == len(
        training_labels
    ), "Not all training sequences were processed."

    logging.info("Starting to process test sequences")
    test_embeddings = embed_dataset(model, test_sequences)
    logging.info("Test sequences processed")
    assert len(test_embeddings) == len(
        test_labels
    ), "Not all test sequences were processed."

    # Function to save embeddings and labels in an HDF5 file, using index as ID
    def save_embeddings_with_index_as_id(filename, embeddings, labels):
        with h5py.File(filename, "w") as f:
            for i, embedding in enumerate(embeddings):
                ds_name = str(i)
                f.create_dataset(ds_name, data=embedding)

                # Convert labels to fixed-length ASCII strings
                encoded_label = np.string_(labels[i])
                f[ds_name].attrs["label"] = encoded_label

    logging.info("Saving embeddings and labels in an HDF5 file")

    # Create a directory path that includes the model and dataset names
    embeddings_dir = f"./embeddings/{model_short_name}_{dataset_name}"
    os.makedirs(embeddings_dir, exist_ok=True)

    # Save the embeddings and labels in an HDF5 file inside the embeddings directory
    save_embeddings_with_index_as_id(
        os.path.join(embeddings_dir, "train.h5py"),
        training_embeddings,
        training_labels,
    )

    save_embeddings_with_index_as_id(
        os.path.join(embeddings_dir, "test.h5py"), test_embeddings, test_labels
    )

    logging.info("Script completed successfully")


if __name__ == "__main__":
    fire.Fire(main)
