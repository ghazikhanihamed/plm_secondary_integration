import os
import h5py
import numpy as np


def load_embeddings_and_labels(embedding_dir, dataset_name):
    train_file = os.path.join(embedding_dir, "train.h5py")
    test_file = os.path.join(embedding_dir, "test.h5py")
    validation_file = os.path.join(embedding_dir, "validation.h5py")

    def load_file(file_path, is_ssp=False):
        embeddings = []
        labels = []
        with h5py.File(file_path, "r") as f:
            for key in f.keys():
                embeddings.append(f[key][()])
                if is_ssp:
                    labels.append(
                        {
                            "label3": f[key].attrs["label3"],
                            "label8": f[key].attrs["label8"],
                            "disorder": f[key].attrs["disorder"],
                        }
                    )
                else:
                    labels.append(f[key].attrs["label"])
        return np.array(embeddings), labels

    train_embeddings, train_labels = load_file(
        train_file, is_ssp=(dataset_name == "SSP")
    )
    test_embeddings, test_labels = load_file(test_file, is_ssp=(dataset_name == "SSP"))

    # Check if validation file exists and combine with training if it does
    if os.path.exists(validation_file):
        validation_embeddings, validation_labels = load_file(
            validation_file, is_ssp=(dataset_name == "SSP")
        )
        # train_embeddings = np.concatenate((train_embeddings, validation_embeddings))
        # train_labels += validation_labels

    return (
        train_embeddings,
        train_labels,
        validation_embeddings,
        validation_labels,
        test_embeddings,
        test_labels,
    )


# Usage example
# train_dataset, test_dataset = load_embeddings_and_labels("./embeddings", "ionchannels")
