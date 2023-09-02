import pandas as pd
from transformers import T5ForSequenceClassification, AutoTokenizer, set_seed
import torch
import wandb
import json
import accelerate
from accelerate.tracking import WandBTracker
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split


# Constants
SEED = 42
WANDB_CONFIG_PATH = "wandb_config.json"
TRAIN_DATASET_PATH = "./dataset/ionchannels_membraneproteins_imbalanced_train.csv"
TEST_DATASET_PATH = "./dataset/ionchannels_membraneproteins_imbalanced_test.csv"
TOOT_PLM_P2S_MODEL_NAME = "ghazikhanihamed/TooT-PLM-P2S"
ANKH_LARGE_MODEL_NAME = "ElnaggarLab/ankh-base"


def load_wandb_config(path):
    """Loads the Weights & Biases configuration from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["api_key"]


def get_model_and_tokenizer(model_name, accelerator):
    """Loads the model, puts it on the device and in eval mode, and retrieves the tokenizer."""
    model = (
        T5ForSequenceClassification.from_pretrained(model_name)
        .to(accelerator.device)
        .eval()
    )  # Use T5ForSequenceClassification
    model = accelerator.prepare(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_embeddings(model, tokenizer, sequences, device, batch_size=1):
    """Extracts embeddings from a model given protein sequences."""
    all_embeddings = []

    # Check if the model is wrapped in DistributedDataParallel
    inner_model = model.module if hasattr(model, "module") else model

    num_batches = len(sequences) // batch_size + (len(sequences) % batch_size != 0)
    for batch_num, idx in enumerate(range(0, len(sequences), batch_size)):
        batch_sequences = [list(seq) for seq in sequences[idx : idx + batch_size]]
        outputs = tokenizer.batch_encode_plus(
            batch_sequences,
            add_special_tokens=True,  # Ensure special tokens are added
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding="max_length",  # Add this line for padding, especially if using batches
        )

        # Replicating encoder input_ids for the decoder
        outputs["decoder_input_ids"] = outputs["input_ids"].clone()
        outputs = {k: v.to(device) for k, v in outputs.items()}

        with torch.no_grad():
            # Get the hidden states (not the logits)
            model_outputs = inner_model.base_model(**outputs)  # Use inner_model here
            embeddings = model_outputs.last_hidden_state

            # Apply max pooling
            pooled_embeddings = torch.max(embeddings, dim=1)[0]

            all_embeddings.append(pooled_embeddings.clone().cpu().detach())

        wandb.log(
            {
                "Current Batch": batch_num + 1,
                "Total Batches": num_batches,
                "Percentage Completed": (batch_num + 1) / num_batches * 100,
            }
        )

    return torch.cat(
        all_embeddings, dim=0
    )  # Now the output shape is (num_samples, embedding_dimension)


def train_and_evaluate(embeddings_train, embeddings_test, y_train, y_test):
    classifiers = {
        "logistic_regression": LogisticRegression(random_state=1),
        "svm": SVC(probability=True),
        "random_forest": RandomForestClassifier(),
        "knn": KNeighborsClassifier(),
        "mlp": MLPClassifier(max_iter=1000),
    }

    for clf_name, clf in classifiers.items():
        # Training the classifier
        clf.fit(embeddings_train, y_train)

        # Predicting and evaluating
        y_pred = clf.predict(embeddings_test)
        y_probas = (
            clf.predict_proba(embeddings_test)
            if hasattr(clf, "predict_proba")
            else None
        )

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        wandb.log({f"{clf_name}_accuracy": accuracy})

        # Compute MCC
        mcc = matthews_corrcoef(y_test, y_pred)
        wandb.log({f"{clf_name}_mcc": mcc})

        # Visualizations
        if y_probas is not None:
            wandb.sklearn.plot_roc(y_test, y_probas, [0, 1], model_name=clf_name)
            wandb.sklearn.plot_precision_recall(
                y_test, y_probas, [0, 1], model_name=clf_name
            )

        wandb.sklearn.plot_confusion_matrix(y_test, y_pred, [0, 1], model_name=clf_name)
        wandb.sklearn.plot_classifier(
            clf,
            embeddings_train,
            embeddings_test,
            y_train,
            y_test,
            y_pred,
            y_probas,
            [0, 1],
            model_name=clf_name,
        )

    # Reporting Summary Metrics
    for clf_name, clf in classifiers.items():
        wandb.sklearn.plot_summary_metrics(
            clf, embeddings_train, y_train, embeddings_test, y_test
        )


def main():
    # Set seed
    set_seed(SEED)

    # Setup Accelerate
    accelerator = accelerate.Accelerator(log_with=["wandb"])
    device = accelerator.device

    with accelerator.main_process_first():
        # Load Weights & Biases Configuration and initialize
        api_key = load_wandb_config(WANDB_CONFIG_PATH)
        wandb.login(key=api_key)

        # Load datasets
        # train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
        # test_dataset = pd.read_csv(TEST_DATASET_PATH)

        # debug
        # Assuming you have loaded the dataset as before
        train_dataset_full = pd.read_csv(TRAIN_DATASET_PATH)
        test_dataset_full = pd.read_csv(TEST_DATASET_PATH)

        # Get a stratified subset of the train dataset
        train_dataset, _ = train_test_split(
            train_dataset_full,
            test_size=0.99,  # Assuming you want 10% of the data, adjust as needed
            stratify=train_dataset_full["label"],  # Stratify according to the labels
            random_state=SEED,
        )

        # Get a stratified subset of the test dataset
        test_dataset, _ = train_test_split(
            test_dataset_full,
            test_size=0.99,  # Assuming you want 10% of the data, adjust as needed
            stratify=test_dataset_full["label"],  # Stratify according to the labels
            random_state=SEED,
        )

        # Setup Weights & Biases
        wandb_tracker = WandBTracker(run_name="plm_secondary_integration")
        accelerator.trackers = [wandb_tracker]

    # Load models and tokenizers
    toot_plm_p2s_model, toot_tokenizer = get_model_and_tokenizer(
        TOOT_PLM_P2S_MODEL_NAME, accelerator
    )
    ankh_large_model, ankh_tokenizer = get_model_and_tokenizer(
        ANKH_LARGE_MODEL_NAME, accelerator
    )

    # Get embeddings
    train_embeddings_toot = get_embeddings(
        toot_plm_p2s_model, toot_tokenizer, train_dataset["sequence"].values, device
    )
    train_embeddings_ankh = get_embeddings(
        ankh_large_model, ankh_tokenizer, train_dataset["sequence"].values, device
    )
    test_embeddings_toot = get_embeddings(
        toot_plm_p2s_model, toot_tokenizer, test_dataset["sequence"].values, device
    )
    test_embeddings_ankh = get_embeddings(
        ankh_large_model, ankh_tokenizer, test_dataset["sequence"].values, device
    )

    # Save embeddings
    torch.save(train_embeddings_toot, "train_embeddings_toot.pt")
    torch.save(train_embeddings_ankh, "train_embeddings_ankh.pt")
    torch.save(test_embeddings_toot, "test_embeddings_toot.pt")
    torch.save(test_embeddings_ankh, "test_embeddings_ankh.pt")

    # Train, predict, and evaluate
    for embeddings_train, embeddings_test, model_name in [
        (train_embeddings_toot, test_embeddings_toot, "toot_plm_p2s"),
        (train_embeddings_ankh, test_embeddings_ankh, "ankh_base"),
    ]:
        y_train = train_dataset["label"].values
        y_test = test_dataset["label"].values

        # Train and evaluate multiple classifiers
        train_and_evaluate(
            embeddings_train.cpu().numpy(),
            embeddings_test.cpu().numpy(),
            y_train,
            y_test,
        )

    wandb.finish()


if __name__ == "__main__":
    main()
