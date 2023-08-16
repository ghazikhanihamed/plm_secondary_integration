import accelerate
import torch
from transformers import (
    T5EncoderModel,
    AutoTokenizer,
    pipeline,
    set_seed,
)
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import wandb
import json

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
test_dataset = pd.read_csv("./dataset/ionchannels_membraneproteins_imbalanced_test.csv")

accelerator = accelerate.Accelerator(log_with="wandb", mixed_precision="fp16")

toot_plm_p2s_model_name = "ghazikhanihamed/TooT-PLM-P2S"
ankh_large_model_name = "ElnaggarLab/ankh-large"

toot_plm_p2s_model = T5EncoderModel.from_pretrained(toot_plm_p2s_model_name)
ankh_large_model = T5EncoderModel.from_pretrained(ankh_large_model_name)

tokenizer = AutoTokenizer.from_pretrained(ankh_large_model_name)

# Convert the models and tokenizer to accelerate's format
toot_plm_p2s_model, tokenizer_toot = accelerator.prepare(toot_plm_p2s_model, tokenizer)
ankh_large_model, tokenizer_ankh = accelerator.prepare(ankh_large_model, tokenizer)

feature_extraction_toot = pipeline(
    task="feature-extraction",
    model=toot_plm_p2s_model,
    tokenizer=tokenizer_toot,
    device_map="auto",
)

feature_extraction_ankh = pipeline(
    task="feature-extraction",
    model=ankh_large_model,
    tokenizer=tokenizer_ankh,
    device_map="auto",
)


def get_embeddings(pipeline, protein_sequences, batch_size=1):
    all_embeddings = []
    num_sequences = len(protein_sequences)
    log_interval = max(
        1, num_sequences // (10 * batch_size)
    )  # Let's log progress every 10% of the dataset

    for i in range(0, num_sequences, batch_size):
        batch_sequences = protein_sequences[i : i + batch_size].tolist()
        embeddings = pipeline(batch_sequences)  # Might return a list

        # Convert embeddings to tensor if it's a list
        if isinstance(embeddings, list):
            embeddings = torch.tensor(embeddings).to(accelerator.device)

        max_pooled = embeddings.max(dim=1)[
            0
        ]  # Max pooling over sequence length. Shape: [batch_size, embedding_dim]

        # Move max_pooled tensor to the correct device before appending
        all_embeddings.append(max_pooled.to(accelerator.device))

        # Log to wandb
        if (i // batch_size) % log_interval == 0:
            percentage_done = (i / num_sequences) * 100
            wandb.log({"percentage_processed": percentage_done})

    return torch.cat(all_embeddings, dim=0)


print(toot_plm_p2s_model.device)
print(ankh_large_model.device)

train_embeddings_toot = get_embeddings(
    feature_extraction_toot, train_dataset["sequence"].values
)
train_embeddings_ankh = get_embeddings(
    feature_extraction_ankh, train_dataset["sequence"].values
)

test_embeddings_toot = get_embeddings(
    feature_extraction_toot, test_dataset["sequence"].values
)
test_embeddings_ankh = get_embeddings(
    feature_extraction_ankh, test_dataset["sequence"].values
)

# Saving embeddings
torch.save(train_embeddings_toot, "train_embeddings_toot.pt")
torch.save(train_embeddings_ankh, "train_embeddings_ankh.pt")
torch.save(test_embeddings_toot, "test_embeddings_toot.pt")
torch.save(test_embeddings_ankh, "test_embeddings_ankh.pt")

train_embeddings_toot_np = train_embeddings_toot.cpu().numpy()
train_embeddings_ankh_np = train_embeddings_ankh.cpu().numpy()

test_embeddings_toot_np = test_embeddings_toot.cpu().numpy()
test_embeddings_ankh_np = test_embeddings_ankh.cpu().numpy()


# Train logistic regressions
lr_toot = LogisticRegression(random_state=1).fit(
    train_embeddings_toot_np, train_dataset["label"]
)
lr_ankh = LogisticRegression(random_state=1).fit(
    train_embeddings_ankh_np, train_dataset["label"]
)

# Predictions
preds_toot = lr_toot.predict(test_embeddings_toot_np)
preds_ankh = lr_ankh.predict(test_embeddings_ankh_np)

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
