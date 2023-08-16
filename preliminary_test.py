import pandas as pd
from transformers import T5EncoderModel, AutoTokenizer, set_seed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import torch
import wandb
import json
import accelerate


def main():
    # Set seed
    set_seed(42)
    # Load Weights & Biases Configuration
    with open("wandb_config.json") as f:
        data = json.load(f)

    api_key = data["api_key"]

    wandb_config = {
        "project": "plm_secondary_integration",
    }

    wandb.login(key=api_key)

    wandb.init(config=wandb_config)

    train_dataset = pd.read_csv(
        "./dataset/ionchannels_membraneproteins_imbalanced_train.csv"
    )
    test_dataset = pd.read_csv(
        "./dataset/ionchannels_membraneproteins_imbalanced_test.csv"
    )

    accelerator = accelerate.Accelerator(log_with="wandb", mixed_precision="fp16")

    toot_plm_p2s_model_name = "ghazikhanihamed/TooT-PLM-P2S"
    ankh_large_model_name = "ElnaggarLab/ankh-large"

    toot_plm_p2s_model = T5EncoderModel.from_pretrained(toot_plm_p2s_model_name)
    ankh_large_model = T5EncoderModel.from_pretrained(ankh_large_model_name)

    toot_plm_p2s_model.half()
    ankh_large_model.half()

    device = accelerator.device

    toot_plm_p2s_model = toot_plm_p2s_model.to(device)
    ankh_large_model = ankh_large_model.to(device)

    toot_plm_p2s_model = toot_plm_p2s_model.eval()
    ankh_large_model = ankh_large_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(ankh_large_model_name)

    def get_embeddings(model, tokenizer, protein_sequences, batch_size=1):
        # Placeholder list to store embeddings
        all_embeddings = []

        # Create batches
        num_batches = len(protein_sequences) // batch_size + (
            len(protein_sequences) % batch_size != 0
        )
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = start_idx + batch_size

            batch_sequences = [
                list(seq) for seq in protein_sequences[start_idx:end_idx]
            ]
            outputs = tokenizer.batch_encode_plus(
                batch_sequences,
                add_special_tokens=False,
                is_split_into_words=True,
                return_tensors="pt",
            )

            outputs = {k: v.to(device) for k, v in outputs.items()}

            with torch.no_grad():
                model_outputs = model(**outputs)
                embeddings = model_outputs.last_hidden_state

                # Max pooling
                mask_expanded = (
                    outputs["attention_mask"]
                    .unsqueeze(-1)
                    .expand(embeddings.size())
                    .float()
                )
                embeddings_max = torch.max(embeddings * mask_expanded, 1).values

                all_embeddings.append(
                    embeddings_max.clone().cpu().detach()
                )  # deep copy

            # Logging progress to wandb
            wandb.log(
                {
                    "Current Batch": batch_num + 1,
                    "Total Batches": num_batches,
                    "Percentage Completed": (batch_num + 1) / num_batches * 100,
                }
            )

        # Concatenate all the embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings

    # Getting the embeddings
    train_embeddings_toot = get_embeddings(
        toot_plm_p2s_model, tokenizer, train_dataset["sequence"].values
    )
    train_embeddings_ankh = get_embeddings(
        ankh_large_model, tokenizer, train_dataset["sequence"].values
    )

    test_embeddings_toot = get_embeddings(
        toot_plm_p2s_model, tokenizer, test_dataset["sequence"].values
    )
    test_embeddings_ankh = get_embeddings(
        ankh_large_model, tokenizer, test_dataset["sequence"].values
    )

    # Saving embeddings
    torch.save(train_embeddings_toot, "train_embeddings_toot.pt")
    torch.save(train_embeddings_ankh, "train_embeddings_ankh.pt")
    torch.save(test_embeddings_toot, "test_embeddings_toot.pt")
    torch.save(test_embeddings_ankh, "test_embeddings_ankh.pt")

    # Train logistic regressions
    lr_toot = LogisticRegression(random_state=1).fit(
        train_embeddings_toot, train_dataset["label"]
    )
    lr_ankh = LogisticRegression(random_state=1).fit(
        train_embeddings_ankh, train_dataset["label"]
    )

    # Predictions
    preds_toot = lr_toot.predict(test_embeddings_toot)
    preds_ankh = lr_ankh.predict(test_embeddings_ankh)

    # Evaluation
    accuracy_toot = accuracy_score(test_dataset["label"], preds_toot)
    accuracy_ankh = accuracy_score(test_dataset["label"], preds_ankh)

    f1_toot = f1_score(test_dataset["label"], preds_toot, average="macro")
    f1_ankh = f1_score(test_dataset["label"], preds_ankh, average="macro")

    mcc_toot = matthews_corrcoef(test_dataset["label"], preds_toot)
    mcc_ankh = matthews_corrcoef(test_dataset["label"], preds_ankh)

    wandb.log(
        {
            "accuracy_toot": accuracy_toot,
            "accuracy_ankh": accuracy_ankh,
            "f1_toot": f1_toot,
            "f1_ankh": f1_ankh,
            "mcc_toot": mcc_toot,
            "mcc_ankh": mcc_ankh,
        }
    )

    print(f"Accuracy for toot_plm_p2s: {accuracy_toot}")
    print(f"Accuracy for ankh_large: {accuracy_ankh}")
    print(f"F1 score for toot_plm_p2s: {f1_toot}")
    print(f"F1 score for ankh_large: {f1_ankh}")
    print(f"MCC for toot_plm_p2s: {mcc_toot}")
    print(f"MCC for ankh_large: {mcc_ankh}")

    wandb.finish()


if __name__ == "__main__":
    main()
