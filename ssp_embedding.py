import os
import torch
import h5py
import logging
from transformers import T5TokenizerFast, T5EncoderModel, set_seed
from datasets import load_dataset
from tqdm import tqdm
import fire
import numpy as np
import random


def main(
    model_name="./best_model_p2s_integration",
    model_short_name="p2s",
    dataset_name="SSP",
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

    training_dataset = load_dataset(
        "proteinea/SSP", data_files={"train": ["training_hhblits.csv"]}
    )
    casp12_dataset = load_dataset("proteinea/SSP", data_files={"test": ["CASP12.csv"]})
    casp13_dataset = load_dataset("proteinea/SSP", data_files={"test": ["CASP13.csv"]})
    casp14_dataset = load_dataset("proteinea/SSP", data_files={"test": ["CASP14.csv"]})
    ts115_dataset = load_dataset("proteinea/SSP", data_files={"test": ["TS115.csv"]})
    cb513_dataset = load_dataset("proteinea/SSP", data_files={"test": ["CB513.csv"]})

    input_column_name = "input"
    labels_3_column_name = "dssp3"
    labels_8_column_name = "dssp8"
    disorder_column_name = "disorder"
    training_sequences, training_labels_3, training_labels_8, training_disorder = (
        training_dataset["train"][input_column_name],
        training_dataset["train"][labels_3_column_name],
        training_dataset["train"][labels_8_column_name],
        training_dataset["train"][disorder_column_name],
    )

    casp12_sequences, casp12_labels_3, casp12_labels_8, casp12_disorder = (
        casp12_dataset["test"][input_column_name],
        casp12_dataset["test"][labels_3_column_name],
        casp12_dataset["test"][labels_8_column_name],
        casp12_dataset["test"][disorder_column_name],
    )

    casp13_sequences, casp13_labels_3, casp13_labels_8, casp13_disorder = (
        casp13_dataset["test"][input_column_name],
        casp13_dataset["test"][labels_3_column_name],
        casp13_dataset["test"][labels_8_column_name],
        casp13_dataset["test"][disorder_column_name],
    )

    casp14_sequences, casp14_labels_3, casp14_labels_8, casp14_disorder = (
        casp14_dataset["test"][input_column_name],
        casp14_dataset["test"][labels_3_column_name],
        casp14_dataset["test"][labels_8_column_name],
        casp14_dataset["test"][disorder_column_name],
    )

    ts115_sequences, ts115_labels_3, ts115_labels_8, ts115_disorder = (
        ts115_dataset["test"][input_column_name],
        ts115_dataset["test"][labels_3_column_name],
        ts115_dataset["test"][labels_8_column_name],
        ts115_dataset["test"][disorder_column_name],
    )

    cb513_sequences, cb513_labels_3, cb513_labels_8, cb513_disorder = (
        cb513_dataset["test"][input_column_name],
        cb513_dataset["test"][labels_3_column_name],
        cb513_dataset["test"][labels_8_column_name],
        cb513_dataset["test"][disorder_column_name],
    )

    def preprocess_dataset(sequences, labels3, labels8, disorder, max_length=None):
        sequences = ["".join(seq.split()) for seq in sequences]

        if max_length is None:
            max_length = len(max(sequences, key=lambda x: len(x)))

        seqs = [list(seq)[:max_length] for seq in sequences]

        labels3 = ["".join(label.split()) for label in labels3]
        labels8 = ["".join(label.split()) for label in labels8]

        labels3 = [list(label)[:max_length] for label in labels3]
        labels8 = [list(label)[:max_length] for label in labels8]

        disorder = [" ".join(disorder.split()) for disorder in disorder]
        disorder = [disorder.split()[:max_length] for disorder in disorder]

        assert len(seqs) == len(labels3) == len(labels8) == len(disorder)
        return seqs, labels3, labels8, disorder

    def embed_dataset(model, sequences, shift_left=0, shift_right=-1):
        inputs_embedding = []
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
                    embedding = model(input_ids=ids["input_ids"].to(device))[0]
                    embedding = (
                        embedding[0].detach().cpu().numpy()[shift_left:shift_right]
                    )
                    inputs_embedding.append(embedding)
                except Exception as e:
                    logging.error(f"Error processing sequence: {sample} - Error: {e}")
                    continue  # Continue with the next sample
        progress_bar.close()
        return inputs_embedding

    (
        training_sequences,
        training_labels3,
        training_labels8,
        training_disorder,
    ) = preprocess_dataset(
        training_sequences, training_labels_3, training_labels_8, training_disorder
    )

    (
        casp12_sequences,
        casp12_labels3,
        casp12_labels8,
        casp12_disorder,
    ) = preprocess_dataset(
        casp12_sequences, casp12_labels_3, casp12_labels_8, casp12_disorder
    )

    (
        casp13_sequences,
        casp13_labels3,
        casp13_labels8,
        casp13_disorder,
    ) = preprocess_dataset(
        casp13_sequences, casp13_labels_3, casp13_labels_8, casp13_disorder
    )

    (
        casp14_sequences,
        casp14_labels3,
        casp14_labels8,
        casp14_disorder,
    ) = preprocess_dataset(
        casp14_sequences, casp14_labels_3, casp14_labels_8, casp14_disorder
    )

    (
        ts115_sequences,
        ts115_labels3,
        ts115_labels8,
        ts115_disorder,
    ) = preprocess_dataset(
        ts115_sequences, ts115_labels_3, ts115_labels_8, ts115_disorder
    )

    (
        cb513_sequences,
        cb513_labels3,
        cb513_labels8,
        cb513_disorder,
    ) = preprocess_dataset(
        cb513_sequences, cb513_labels_3, cb513_labels_8, cb513_disorder
    )

    logging.info("Starting to process training sequences")
    training_embeddings = embed_dataset(model, training_sequences)
    logging.info("Training sequences processed")
    assert (
        len(training_embeddings) == len(training_labels3) == len(training_labels8)
    ), "Not all training sequences were processed."

    logging.info("Starting to process casp12 sequences")
    casp12_embeddings = embed_dataset(model, casp12_sequences)
    logging.info("CASP12 sequences processed")
    assert (
        len(casp12_embeddings) == len(casp12_labels3) == len(casp12_labels8)
    ), "Not all casp12 sequences were processed."

    logging.info("Starting to process casp13 sequences")
    casp13_embeddings = embed_dataset(model, casp13_sequences)
    logging.info("CASP13 sequences processed")
    assert (
        len(casp13_embeddings) == len(casp13_labels3) == len(casp13_labels8)
    ), "Not all casp13 sequences were processed."

    logging.info("Starting to process casp14 sequences")
    casp14_embeddings = embed_dataset(model, casp14_sequences)
    logging.info("CASP14 sequences processed")
    assert (
        len(casp14_embeddings) == len(casp14_labels3) == len(casp14_labels8)
    ), "Not all casp14 sequences were processed."

    logging.info("Starting to process ts115 sequences")
    ts115_embeddings = embed_dataset(model, ts115_sequences)
    logging.info("TS115 sequences processed")
    assert (
        len(ts115_embeddings) == len(ts115_labels3) == len(ts115_labels8)
    ), "Not all ts115 sequences were processed."

    logging.info("Starting to process cb513 sequences")
    cb513_embeddings = embed_dataset(model, cb513_sequences)
    logging.info("CB513 sequences processed")
    assert (
        len(cb513_embeddings) == len(cb513_labels3) == len(cb513_labels8)
    ), "Not all cb513 sequences were processed."

    # Function to save embeddings and labels in an HDF5 file, using index as ID
    def save_embeddings_with_index_as_id(
        filename, embeddings, labels3, labels8, disorder
    ):
        assert (
            len(embeddings) == len(labels3) == len(labels8) == len(disorder)
        ), "Input lists must be of the same length"

        with h5py.File(filename, "w") as f:
            for i, embedding in enumerate(embeddings):
                # Use the index as the ID for each embedding
                ds_name = str(i)
                f.create_dataset(ds_name, data=embedding)
                # Store the label as an attribute of the dataset
                f[ds_name].attrs["label3"] = np.string_(labels3[i])
                f[ds_name].attrs["label8"] = np.string_(labels8[i])
                f[ds_name].attrs["disorder"] = np.string_(" ".join(disorder[i]))

    logging.info("Saving embeddings and labels in an HDF5 file")

    # Create a directory path that includes the model and dataset names
    embeddings_dir = f"./embeddings/{model_short_name}_{dataset_name}"
    os.makedirs(embeddings_dir, exist_ok=True)

    # Save the embeddings and labels in an HDF5 file inside the embeddings directory
    save_embeddings_with_index_as_id(
        os.path.join(embeddings_dir, "train.h5py"),
        training_embeddings,
        training_labels3,
        training_labels8,
        training_disorder,
    )

    save_embeddings_with_index_as_id(
        os.path.join(embeddings_dir, "casp12.h5py"),
        casp12_embeddings,
        casp12_labels3,
        casp12_labels8,
        casp12_disorder,
    )

    save_embeddings_with_index_as_id(
        os.path.join(embeddings_dir, "casp13.h5py"),
        casp13_embeddings,
        casp13_labels3,
        casp13_labels8,
        casp13_disorder,
    )

    save_embeddings_with_index_as_id(
        os.path.join(embeddings_dir, "casp14.h5py"),
        casp14_embeddings,
        casp14_labels3,
        casp14_labels8,
        casp14_disorder,
    )

    save_embeddings_with_index_as_id(
        os.path.join(embeddings_dir, "ts115.h5py"),
        ts115_embeddings,
        ts115_labels3,
        ts115_labels8,
        ts115_disorder,
    )

    save_embeddings_with_index_as_id(
        os.path.join(embeddings_dir, "cb513.h5py"),
        cb513_embeddings,
        cb513_labels3,
        cb513_labels8,
        cb513_disorder,
    )

    logging.info("Script completed successfully")


if __name__ == "__main__":
    fire.Fire(main)
