from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, concatenate_datasets
import logging
import torch
import numpy as np
import sys

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

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base")

# load the dataset
dataset1 = load_dataset(
    "proteinea/secondary_structure_prediction",
    data_files={"train": ["training_hhblits.csv"]},
)
dataset2 = load_dataset(
    "proteinea/secondary_structure_prediction", data_files={"CASP12": ["CASP12.csv"]}
)
dataset3 = load_dataset(
    "proteinea/secondary_structure_prediction", data_files={"CASP14": ["CASP14.csv"]}
)
dataset4 = load_dataset(
    "proteinea/secondary_structure_prediction", data_files={"CB513": ["CB513.csv"]}
)
dataset5 = load_dataset(
    "proteinea/secondary_structure_prediction", data_files={"TS115": ["TS115.csv"]}
)

# concatenate dataset1 and dataset4
train_dataset = concatenate_datasets([dataset1["train"], dataset4["CB513"]])

# The validation set will be dataset5
validation_dataset = dataset5["TS115"]

# Two separate test datasets
test_dataset1 = dataset2["CASP12"]
test_dataset2 = dataset3["CASP14"]

input_column_name = "input"
labels_column_name = "dssp3"

# concatenate all sequences
all_sequences = list(train_dataset[input_column_name])
sequence_lengths = [len(seq) for seq in all_sequences]
max_length = int(np.percentile(sequence_lengths, 99))

print("Max length: ", max_length)

# Consider each label as a tag for each token
unique_tags = set(tag for doc in train_dataset[labels_column_name] for tag in doc)

# add padding tag
unique_tags.add("<pad>")


tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}


def preprocess_data(examples):
    sequences, labels = examples["input"], examples["dssp3"]

    # remove whitespace and split each sequence into list of amino acids
    sequences = [list("".join(seq.split())) for seq in sequences]
    labels = [list("".join(label.split())) for label in labels]

    print("Max length: ", max_length)

    # encode sequences
    inputs = tokenizer(
        sequences,
        add_special_tokens=True,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        is_split_into_words=True,
        return_tensors="pt",
    )

    # encode labels
    labels_encoded = [[tag2id[tag] for tag in label] for label in labels]

    # Pad or truncate the labels to match the sequence length
    labels_encoded = [
        label[:max_length] + [tag2id["<pad>"]] * (max_length - len(label))
        for label in labels_encoded
    ]

    assert len(inputs["input_ids"]) == len(labels_encoded)

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": torch.tensor(labels_encoded, dtype=torch.long),
    }


train_dataset = train_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Running tokenizer on dataset for training",
)

valid_dataset = validation_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=validation_dataset.column_names,
    desc="Running tokenizer on dataset for validation",
)


def q3_accuracy(y_true, y_pred):
    """
    Computes Q3 accuracy.
    y_true: List of actual values
    y_pred: List of predicted values
    """
    y_true_flat = [tag for seq in y_true for tag in seq if tag != tag2id["<pad>"]]
    y_pred_flat = [tag for seq in y_pred for tag in seq if tag != tag2id["<pad>"]]
    correct = sum(t == p for t, p in zip(y_true_flat, y_pred_flat))
    return correct / len(y_true_flat)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[0] if isinstance(predictions, tuple) else predictions
    # convert numpy ndarray to Tensor
    predictions = torch.tensor(predictions)
    predictions = torch.argmax(predictions, dim=-1)
    return {"q3_accuracy": q3_accuracy(labels.tolist(), predictions.tolist())}


# Create the model and prepare it
model = T5ForConditionalGeneration.from_pretrained("ElnaggarLab/ankh-base")

experiment = "p2s"

# Prepare training args
training_args = TrainingArguments(
    output_dir=f"./results_{experiment}",
    num_train_epochs=20,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=1000,
    learning_rate=1e-03,
    weight_decay=0.0,
    logging_dir=f"./logs_{experiment}",
    logging_steps=200,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=4,
    fp16=False,
    fp16_opt_level="02",
    seed=7,
    load_best_model_at_end=True,
    metric_for_best_model="eval_q3_accuracy",
    greater_is_better=True,
    save_strategy="epoch",
    remove_unused_columns=False,
    run_name="SS-Generation",
    report_to="wandb",
    weight_decay=1e-5,
    hub_token="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",
    hub_model_id="ghazikhanihamed/TooT-PLM-P2S",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# We save the best model in the folder "best_model"
trainer.save_model("./best_model")

# Push model to hub
trainer.push_to_hub(commit_message="PLM-Secondary-Structure-Generation")


# Evaluate the model on test datasets
test_dataset1 = test_dataset1.map(
    preprocess_data,
    batched=True,
    remove_columns=test_dataset1.column_names,
    desc="Running tokenizer on dataset for test",
)

test_dataset2 = test_dataset2.map(
    preprocess_data,
    batched=True,
    remove_columns=test_dataset2.column_names,
    desc="Running tokenizer on dataset for test",
)
# Set test dataset1 as evaluation dataset and evaluate
trainer.eval_dataset = test_dataset1
metrics_test1 = trainer.evaluate()

# Set test dataset2 as evaluation dataset and evaluate
trainer.eval_dataset = test_dataset2
metrics_test2 = trainer.evaluate()

print("Evaluation results on test set CASP12: ", metrics_test1)
print("Evaluation results on test set CASP14: ", metrics_test2)
