from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
)
from accelerate import Accelerator
from datasets import load_dataset
import torch
import numpy as np

tokenizer = T5Tokenizer.from_pretrained("ElnaggarLab/ankh-large")

# load the dataset
training_dataset = load_dataset(
    "proteinea/secondary_structure_prediction",
    data_files={"train": ["training_hhblits.csv"]},
)
casp12_dataset = load_dataset("proteinea/SSP", data_files={"test": ["CASP12.csv"]})

input_column_name = "input"
labels_column_name = "dssp3"

# concatenate all sequences
all_sequences = list(training_dataset["train"][input_column_name]) + list(
    casp12_dataset["test"][input_column_name]
)

sequence_lengths = [len(seq.split()) for seq in all_sequences]
max_length = int(np.percentile(sequence_lengths, 95))

# Consider each label as a tag for each token
unique_tags = set(
    tag for doc in training_dataset["train"][labels_column_name] for tag in doc
)
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

    # truncate and pad labels to match the sequence length
    labels_encoded = [
        [tag2id[tag] for tag in label] + [tag2id["<pad>"]] * (max_length - len(label))
        for label in labels
    ]

    assert len(inputs["input_ids"]) == len(labels_encoded)

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels_encoded,
    }


# Process the datasets
train_dataset = training_dataset.map(
    preprocess_data, batched=True, remove_columns=training_dataset.column_names["train"]
)
valid_dataset = casp12_dataset.map(
    preprocess_data, batched=True, remove_columns=casp12_dataset.column_names["test"]
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
    # or your own way of getting predictions
    predictions = torch.argmax(predictions, dim=-1)
    return {"q3_accuracy": q3_accuracy(labels.tolist(), predictions.tolist())}


# Prepare the model
model = T5ForConditionalGeneration.from_pretrained("ElnaggarLab/ankh-large")

# Prepare training args
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=50,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=100,
    logging_dir="./logs",
    logging_steps=1000,
    evaluation_strategy="epoch",
    fp16=True,
    deepspeed="ds_config.json",
)

# Initialize accelerator
accelerator = Accelerator()

# Apply data and model parallelism, if specified in the deepspeed configuration
model, train_dataset, valid_dataset, training_args = accelerator.prepare(
    model, train_dataset, valid_dataset, training_args
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
