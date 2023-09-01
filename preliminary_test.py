import pandas as pd
from transformers import T5EncoderModel, AutoTokenizer, set_seed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import torch
import wandb
import json
import accelerate

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
    model = T5EncoderModel.from_pretrained(model_name).to(accelerator.device).eval()
    model = accelerator.prepare(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_embeddings(model, tokenizer, sequences, device, batch_size=1):
    """Extracts embeddings from a model given protein sequences."""
    all_embeddings = []

    num_batches = len(sequences) // batch_size + (len(sequences) % batch_size != 0)
    for batch_num, idx in enumerate(range(0, len(sequences), batch_size)):
        batch_sequences = [list(seq) for seq in sequences[idx : idx + batch_size]]
        outputs = tokenizer.batch_encode_plus(
            batch_sequences,
            add_special_tokens=False,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        outputs = {k: v.to(device) for k, v in outputs.items()}

        with torch.no_grad():
            model_outputs = model(**outputs)
            embeddings = model_outputs.last_hidden_state

            mask_expanded = (
                outputs["attention_mask"]
                .unsqueeze(-1)
                .expand(embeddings.size())
                .float()
            )
            embeddings_max = torch.max(embeddings * mask_expanded, 1).values

            all_embeddings.append(embeddings_max.clone().cpu().detach())

        wandb.log(
            {
                "Current Batch": batch_num + 1,
                "Total Batches": num_batches,
                "Percentage Completed": (batch_num + 1) / num_batches * 100,
            }
        )

    return torch.cat(all_embeddings, dim=0)


def main():
    # Set seed
    set_seed(SEED)

    # Load Weights & Biases Configuration and initialize
    api_key = load_wandb_config(WANDB_CONFIG_PATH)
    wandb.login(key=api_key)
    wandb.init(config={"project": "plm_secondary_integration"})

    # Load datasets
    train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    test_dataset = pd.read_csv(TEST_DATASET_PATH)

    # Setup Accelerate
    accelerator = accelerate.Accelerator(log_with="wandb")
    device = accelerator.device

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

    # Train, predict and evaluate
    for embeddings, model_name in [
        (train_embeddings_toot, "toot_plm_p2s"),
        (train_embeddings_ankh, "ankh_base"),
    ]:
        lr = LogisticRegression(random_state=1).fit(embeddings, train_dataset["label"])
        preds = lr.predict(
            test_embeddings_toot
            if model_name == "toot_plm_p2s"
            else test_embeddings_ankh
        )

        accuracy = accuracy_score(test_dataset["label"], preds)
        f1 = f1_score(test_dataset["label"], preds, average="macro")
        mcc = matthews_corrcoef(test_dataset["label"], preds)

        if accelerator.is_main_process:
            wandb.log(
                {
                    f"accuracy_{model_name}": accuracy,
                    f"f1_{model_name}": f1,
                    f"mcc_{model_name}": mcc,
                }
            )

        accelerator.print(f"Accuracy for {model_name}: {accuracy}")
        accelerator.print(f"F1 score for {model_name}: {f1}")
        accelerator.print(f"MCC for {model_name}: {mcc}")

    wandb.finish()


if __name__ == "__main__":
    main()
