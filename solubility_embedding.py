import os
import torch
import numpy as np
import random
import h5py
from transformers import T5TokenizerFast, T5EncoderModel
from tqdm.auto import tqdm
from datasets import load_dataset

# Set seeds and device
seed = 7
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model = T5EncoderModel.from_pretrained("./best_model_p2s_integration").to(device).eval()
tokenizer = T5TokenizerFast.from_pretrained("./best_model_p2s_integration")

# Load dataset
dataset = load_dataset("proteinea/solubility")

# Merge training and validation sets
train_val_sequences = dataset["train"]["sequences"] + dataset["validation"]["sequences"]
train_val_labels = dataset["train"]["labels"] + dataset["validation"]["labels"]
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
        max_length = len(max(train_val_sequences, key=lambda x: len(x)))
    splitted_sequences = [list(seq[:max_length]) for seq in sequences]
    return splitted_sequences, labels


# Embedding function
def embed_dataset(model, sequences, shift_left=0, shift_right=-1):
    embeddings = []
    with torch.no_grad():
        for sample in tqdm(sequences):
            ids = tokenizer.batch_encode_plus(
                [sample],
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
            embedding = model(input_ids=ids["input_ids"].to(device))[0]
            embedding = embedding[0].detach().cpu().numpy()[shift_left:shift_right]
            embeddings.append(embedding)
    return embeddings


# Preprocess datasets
train_val_sequences, train_val_labels = preprocess_dataset(
    train_val_sequences, train_val_labels
)
test_sequences, test_labels = preprocess_dataset(test_sequences, test_labels)

# Embed datasets
train_val_embeddings = embed_dataset(model, train_val_sequences)
test_embeddings = embed_dataset(model, test_sequences)


# Save embeddings with labels using HDF5
def save_embeddings_hdf5(embeddings, labels, file_name):
    with h5py.File(file_name, "w") as h5f:
        h5f.create_dataset("embeddings", data=np.array(embeddings))
        h5f.create_dataset("labels", data=np.array(labels))


# Save data
model_name = "p2s"
dataset_name = "solubility"
os.makedirs("./embeddings", exist_ok=True)
save_embeddings_hdf5(
    train_val_embeddings,
    train_val_labels,
    f"./embeddings/{model_name}_{dataset_name}_train.h5",
)
save_embeddings_hdf5(
    test_embeddings, test_labels, f"./embeddings/{model_name}_{dataset_name}_test.h5"
)
