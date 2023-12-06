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
        return embeddings, labels

    train_embeddings, train_labels = load_file(
        train_file, is_ssp=(dataset_name == "SSP")
    )
    test_embeddings, test_labels = load_file(test_file, is_ssp=(dataset_name == "SSP"))

    # Check if validation file exists and combine with training if it does
    if os.path.exists(validation_file):
        validation_embeddings, validation_labels = load_file(
            validation_file, is_ssp=(dataset_name == "SSP")
        )
        return (
            train_embeddings,
            train_labels,
            validation_embeddings,
            validation_labels,
            test_embeddings,
            test_labels,
        )
        # train_embeddings = np.concatenate((train_embeddings, validation_embeddings))
        # train_labels += validation_labels

    return train_embeddings, train_labels, test_embeddings, test_labels


def load_ssp_embeddings_and_labels(embedding_dir):
    train_file = os.path.join(embedding_dir, "train.h5py")
    casp12_file = os.path.join(embedding_dir, "casp12.h5py")
    casp13_file = os.path.join(embedding_dir, "casp13.h5py")
    casp14_file = os.path.join(embedding_dir, "casp14.h5py")
    ts115_file = os.path.join(embedding_dir, "ts115.h5py")
    cb513_file = os.path.join(embedding_dir, "cb513.h5py")

    def load_file(file_path):
        embeddings = []
        labels = []
        with h5py.File(file_path, "r") as f:
            for key in f.keys():
                embeddings.append(f[key][()])
                labels.append(
                    {
                        "label3": f[key].attrs["label3"],
                        "label8": f[key].attrs["label8"],
                        "disorder": f[key].attrs["disorder"],
                    }
                )
        return embeddings, labels

    train_embeddings, train_labels = load_file(train_file)
    casp12_embeddings, casp12_labels = load_file(casp12_file)
    casp13_embeddings, casp13_labels = load_file(casp13_file)
    casp14_embeddings, casp14_labels = load_file(casp14_file)
    ts115_embeddings, ts115_labels = load_file(ts115_file)
    cb513_embeddings, cb513_labels = load_file(cb513_file)

    return (
        train_embeddings,
        train_labels,
        casp12_embeddings,
        casp12_labels,
        casp13_embeddings,
        casp13_labels,
        casp14_embeddings,
        casp14_labels,
        ts115_embeddings,
        ts115_labels,
        cb513_embeddings,
        cb513_labels,
    )


# Usage example
# train_dataset, test_dataset = load_embeddings_and_labels("./embeddings", "ionchannels")
