from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)
from datasets import load_dataset
from transformers.trainer_utils import set_seed
import logging
import torch
import numpy as np
import sys


logger = logging.getLogger(__name__)

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
training_dataset = load_dataset(
    "proteinea/secondary_structure_prediction",
    data_files={"train": ["training_hhblits.csv"]},
)
casp12_dataset = load_dataset(
    "proteinea/secondary_structure_prediction", data_files={"test": ["CASP12.csv"]}
)

casp14_dataset = load_dataset(
    "proteinea/secondary_structure_prediction", data_files={"test": ["CASP14.csv"]}
)

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


train_dataset = training_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=training_dataset.column_names["train"],
    desc="Running tokenizer on dataset",
)
valid_dataset = casp12_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=casp12_dataset.column_names["test"],
    desc="Running tokenizer on dataset",
)
test_dataset = casp14_dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=casp14_dataset.column_names["test"],
    desc="Running tokenizer on dataset",
)


def q3_accuracy(y_true, y_pred):
    """
    Computes Q3 accuracy.
    y_true: List of actual values
    y_pred: List of predicted values
    """
    y_true_flat = [
        tag for seq in y_true for tag in seq if tag != tag2id["<pad>"]]
    y_pred_flat = [
        tag for seq in y_pred for tag in seq if tag != tag2id["<pad>"]]
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
    learning_rate=3e-4,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir="./logs",
    logging_strategy="epoch",
    save_strategy="epoch",
    deepspeed="ds_config.json",
    load_best_model_at_end=True,
    metric_for_best_model="q3_accuracy",
    greater_is_better=True,
    num_train_epochs=20,
    fp16=True,
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

# Evaluate the model
trainer.evaluate()
