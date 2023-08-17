import pandas as pd

import wandb
import json

# Load Weights & Biases Configuration
with open("wandb_config.json") as f:
    data = json.load(f)

api_key = data["api_key"]

wandb_config = {
    "project": "plm_secondary_integration",
}

wandb.login(key=api_key)

wandb.init(config=wandb_config)

train_df = pd.read_csv("./dataset/ionchannels_membraneproteins_imbalanced_train.csv")
test_df = pd.read_csv("./dataset/ionchannels_membraneproteins_imbalanced_test.csv")

train_texts = train_df["sequence"].tolist()
train_labels = (
    train_df["label"].apply(lambda x: 0 if x == "ionchannels" else 1).tolist()
)

test_texts = test_df["sequence"].tolist()
test_labels = test_df["label"].apply(lambda x: 0 if x == "ionchannels" else 1).tolist()

from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.1, stratify=train_labels
)

train_texts = ["classify: " + text for text in train_texts]
test_texts = ["classify: " + text for text in test_texts]

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ghazikhanihamed/TooT-PLM-P2S")
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


val_texts = ["classify: " + text for text in val_texts]
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

import torch


class IonDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = IonDataset(train_encodings, train_labels)
test_dataset = IonDataset(test_encodings, test_labels)
val_dataset = IonDataset(val_encodings, val_labels)

from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments

model = T5ForConditionalGeneration.from_pretrained("ghazikhanihamed/TooT-PLM-P2S")

from sklearn.metrics import matthews_corrcoef


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=1)
    return {"mcc": matthews_corrcoef(labels, preds)}


training_args = TrainingArguments(
    output_dir="./results",
    do_train=True,
    do_eval=True,
    deepspeed="./ds_config_finetune.json",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_mcc",
    greater_is_better=True,
    num_train_epochs=20,
    seed=42,
    run_name="SS-Generation",
    report_to="wandb",
    gradient_accumulation_steps=1,
    learning_rate=1e-6,
    fp16=False,
    remove_unused_columns=False,
    hub_token="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",
    push_to_hub=True,
    hub_model_id="ghazikhanihamed/TooT-PLM-P2S_ionchannels-membrane",
)


trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Switch to test dataset for evaluation
trainer.eval_dataset = test_dataset
trainer.evaluate()


def extract_encoder_representations(model, encodings):
    with torch.no_grad():
        # Forward pass to get encoder's final hidden states
        encoder_output = model.get_encoder()(
            input_ids=encodings["input_ids"], attention_mask=encodings["attention_mask"]
        )
        # Max pooling across the sequence length dimension
        pooled_output, _ = encoder_output.last_hidden_state.max(dim=1)
        return pooled_output.numpy()


train_representations = extract_encoder_representations(trainer.model, train_encodings)
test_representations = extract_encoder_representations(trainer.model, test_encodings)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

clf = SVC()
clf.fit(train_representations, train_labels)

train_preds = clf.predict(train_representations)
test_preds = clf.predict(test_representations)

print("Train Accuracy:", accuracy_score(train_labels, train_preds))
print("Test Accuracy:", accuracy_score(test_labels, test_preds))

print("Train F1:", f1_score(train_labels, train_preds))
print("Test F1:", f1_score(test_labels, test_preds))

print("Train MCC:", matthews_corrcoef(train_labels, train_preds))
print("Test MCC:", matthews_corrcoef(test_labels, test_preds))
