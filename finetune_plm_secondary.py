import pandas as pd
import wandb
import json
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from sklearn.svm import SVC
import numpy as np

# Load Weights & Biases Configuration
with open("wandb_config.json") as f:
    data = json.load(f)
api_key = data["api_key"]

# Weights & Biases Configuration
wandb_config = {"project": "plm_secondary_integration"}
wandb.login(key=api_key)
wandb.init(config=wandb_config)

# Load dataset
train_df = pd.read_csv("./dataset/ionchannels_membraneproteins_imbalanced_train.csv")
test_df = pd.read_csv("./dataset/ionchannels_membraneproteins_imbalanced_test.csv")

# Process dataset
train_texts = train_df["sequence"].tolist()
train_labels = (
    train_df["label"].apply(lambda x: 1 if x == "ionchannels" else 0).tolist()
)
test_texts = test_df["sequence"].tolist()
test_labels = test_df["label"].apply(lambda x: 1 if x == "ionchannels" else 0).tolist()

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.1, stratify=train_labels
)

# Preprocess text
train_texts = ["classify: " + text for text in train_texts]
test_texts = ["classify: " + text for text in test_texts]
val_texts = ["classify: " + text for text in val_texts]

# For the second code, labels are binary. Therefore, we use only two 'tags': 0 and 1.
tag2id = {0: 0, 1: 1}
id2tag = {0: 0, 1: 1}

all_sequences = train_texts
sequence_lengths = [len(seq.replace("classify: ", "")) for seq in all_sequences]
max_length = int(np.percentile(sequence_lengths, 95))

tokenizer = AutoTokenizer.from_pretrained("ghazikhanihamed/TooT-PLM-P2S")


def preprocess_data(examples):
    sequences, labels = examples["sequence"], examples["label"]

    sequences = [list(seq) for seq in sequences]  # Convert to list of characters

    # encode sequences
    inputs = tokenizer.batch_encode_plus(
        sequences,
        add_special_tokens=True,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        is_split_into_words=True,
        return_tensors="pt",
    )

    assert len(inputs["input_ids"]) == len(labels)

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
    }


train_data = {"sequence": train_texts, "label": train_labels}
val_data = {"sequence": val_texts, "label": val_labels}
test_data = {"sequence": test_texts, "label": test_labels}

train_processed = preprocess_data(train_data)
val_processed = preprocess_data(val_data)
test_processed = preprocess_data(test_data)


class IonDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        item = {
            "input_ids": self.data["input_ids"][idx],
            "attention_mask": self.data["attention_mask"][idx],
            "labels": self.data["labels"][idx],
        }
        return item

    def __len__(self):
        return len(self.data["labels"])


train_dataset = IonDataset(train_processed)
test_dataset = IonDataset(test_processed)
val_dataset = IonDataset(val_processed)


print(train_processed["input_ids"].shape)
print(val_processed["input_ids"].shape)
print(test_processed["input_ids"].shape)


# Load pre-trained model
model = T5ForConditionalGeneration.from_pretrained("ghazikhanihamed/TooT-PLM-P2S")

sample_input = train_processed["input_ids"][:2]  # Take a small batch for testing
sample_output = model(input_ids=sample_input, decoder_input_ids=sample_input)


# Metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=1)
    return {"mcc": matthews_corrcoef(labels, preds)}


# Training arguments
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
    gradient_accumulation_steps=32,
    learning_rate=1e-6,
    max_grad_norm=1.0,
    fp16=False,
    hub_token="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",
    push_to_hub=True,
    hub_model_id="ghazikhanihamed/TooT-PLM-P2S_ionchannels-membrane",
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,  # Pass the tokenizer
)

# Train the model
trainer.train()

# Evaluate on the test dataset
trainer.eval_dataset = test_dataset
trainer.evaluate()


# Function to extract encoder representations
def extract_encoder_representations(model, encodings):
    with torch.no_grad():
        encoder_output = model.get_encoder()(
            input_ids=encodings["input_ids"], attention_mask=encodings["attention_mask"]
        )
        pooled_output, _ = encoder_output.last_hidden_state.max(dim=1)
        return pooled_output.cpu().numpy()  # Convert tensor to numpy array


# Extract encoder representations for train and test data
train_representations = extract_encoder_representations(trainer.model, train_processed)
test_representations = extract_encoder_representations(trainer.model, test_processed)


# Train an SVM classifier
clf = SVC()
clf.fit(train_representations, train_labels)

# Predictions
train_preds = clf.predict(train_representations)
test_preds = clf.predict(test_representations)

# Calculate and print evaluation metrics
print("Train Accuracy:", accuracy_score(train_labels, train_preds))
print("Test Accuracy:", accuracy_score(test_labels, test_preds))
print("Train F1:", f1_score(train_labels, train_preds))
print("Test F1:", f1_score(test_labels, test_preds))
print("Train MCC:", matthews_corrcoef(train_labels, train_preds))
print("Test MCC:", matthews_corrcoef(test_labels, test_preds))
