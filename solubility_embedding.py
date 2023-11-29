import os
import torch
import numpy as np
import random
import h5py
import logging
from transformers import T5TokenizerFast, T5EncoderModel
from datasets import load_dataset
from tqdm.auto import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Uncomment the following line to log to a file
# logging.basicConfig(filename='process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting script")

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


seed = 7
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Available device: {device}")

model_name = "./best_model_p2s_integration"
model = T5EncoderModel.from_pretrained(model_name).to(device).eval()
tokenizer = T5TokenizerFast.from_pretrained(model_name)


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


model.eval()
model.to(device=device)
logging.info(f"Number of parameters: {get_num_params(model)}")

dataset = load_dataset("proteinea/solubility")
logging.info("Dataset loaded")

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
            embedding = model(input_ids=ids["input_ids"].to(device))[0]
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

logging.info("Starting to process training sequences")
training_embeddings = embed_dataset(model, training_sequences)
logging.info("Training sequences processed")

logging.info("Starting to process validation sequences")
validation_embeddings = embed_dataset(model, validation_sequences)
logging.info("Validation sequences processed")

logging.info("Starting to process test sequences")
test_embeddings = embed_dataset(model, test_sequences)
logging.info("Test sequences processed")


# Concatenate training and validation sets
combined_train_embeddings = np.concatenate(
    (training_embeddings, validation_embeddings), axis=0
)
combined_train_labels = training_labels + validation_labels


# Function to save embeddings and labels in an HDF5 file, using index as ID
def save_embeddings_with_index_as_id(filename, embeddings, labels):
    with h5py.File(filename, "w") as f:
        for i, embedding in enumerate(embeddings):
            # Use the index as the ID for each embedding
            dataset_name = str(i)
            f.create_dataset(dataset_name, data=embedding)
            # Store the label as an attribute of the dataset
            f[dataset_name].attrs["label"] = labels[i]


logging.info("Saving embeddings and labels in an HDF5 file")

model_short_name = "p2s"
dataset_name = "solubility"

# Create a directory path that includes the model and dataset names
embeddings_dir = f"./embeddings/{model_short_name}_{dataset_name}"
os.makedirs(embeddings_dir, exist_ok=True)

# Save the embeddings and labels in an HDF5 file inside the embeddings directory
save_embeddings_with_index_as_id(
    os.path.join(embeddings_dir, "train.h5py"),
    combined_train_embeddings,
    combined_train_labels,
)
save_embeddings_with_index_as_id(
    os.path.join(embeddings_dir, "test.h5py"), test_embeddings, test_labels
)

logging.info("Script completed successfully")
