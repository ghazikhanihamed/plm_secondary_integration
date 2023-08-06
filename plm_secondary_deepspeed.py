from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)
from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets
from transformers.trainer_utils import set_seed
import logging
import torch
import numpy as np
import sys

import wandb
import json

# Load Weights & Biases Configuration
with open("wandb_config.json") as f:
    data = json.load(f)

api_key = data["wandb"]["api_key"]

wandb_config = {
    "project": "plm_secondary_integration",
}

wandb.login(key=api_key)

wandb.init(config=wandb_config)

accelerator = Accelerator()

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Set seed before initializing model.
set_seed(42)

tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-large")

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

if accelerator.is_main_process:
    # Print the number of samples
    accelerator.print(f"Number of training samples: {len(train_dataset)}")
    accelerator.print(f"Number of validation samples: {len(validation_dataset)}")
    accelerator.print(f"Number of test samples on CASP12: {len(test_dataset1)}")
    accelerator.print(f"Number of test samples on CASP14: {len(test_dataset2)}")
accelerator.wait_for_everyone()

input_column_name = "input"
labels_column_name = "dssp3"

# concatenate all sequences
all_sequences = list(train_dataset[input_column_name])


sequence_lengths = [len(seq.split()) for seq in all_sequences]
max_length = int(np.percentile(sequence_lengths, 95))

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


with accelerator.main_process_first():
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
    return {"eval_q3_accuracy": q3_accuracy(labels.tolist(), predictions.tolist())}


train_dataset, valid_dataset, test_dataset1, test_dataset2 = accelerator.prepare(
    train_dataset, valid_dataset, test_dataset1, test_dataset2
)

# Create the model and prepare it
model = T5ForConditionalGeneration.from_pretrained("ElnaggarLab/ankh-large")
model = accelerator.prepare(model)


# Prepare training args
training_args = TrainingArguments(
    output_dir="./results",
    do_train=True,
    do_eval=True,
    deepspeed="./ds_config.json",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_q3_accuracy",
    greater_is_better=True,
    num_train_epochs=20,
    save_total_limit=1,
    seed=42,
    run_name="SS-Generation",
    report_to="wandb",
    gradient_accumulation_steps=1,
    learning_rate=3e-5,
    weight_decay=3e-7,
    adam_beta1=0.8,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    warmup_steps=500,
    fp16=True,
    remove_unused_columns=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./results")

# Evaluate the model on test datasets
# Set test dataset1 as evaluation dataset and evaluate
trainer.eval_dataset = test_dataset1
metrics_test1 = trainer.evaluate()

# Set test dataset2 as evaluation dataset and evaluate
trainer.eval_dataset = test_dataset2
metrics_test2 = trainer.evaluate()

print("Evaluation results on test set CASP12: ", metrics_test1)
print("Evaluation results on test set CASP14: ", metrics_test2)
