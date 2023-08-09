import pandas as pd
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
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

toot_plm_p2s_model_name = "ghazikhanihamed/TooT-PLM-P2S"
ankh_large_model_name = "ElnaggarLab/ankh-large"

toot_plm_p2s_model = AutoModel.from_pretrained(toot_plm_p2s_model_name)
ankh_large_model = AutoModel.from_pretrained(ankh_large_model_name)

toot_plm_p2s_model.eval()
ankh_large_model.eval()

# Initialize DeepSpeed-Inference for each model

toot_plm_p2s_model = deepspeed.init_inference(
    toot_plm_p2s_model, config="ds_config_inference.json")
ankh_large_model = deepspeed.init_inference(
    ankh_large_model, config="ds_config_inference.json")

tokenizer = AutoTokenizer.from_pretrained(ankh_large_model_name)

train_dataset = pd.read_csv(
    "./dataset/ionchannels_membraneproteins_imbalanced_train.csv")
test_dataset = pd.read_csv(
    "./dataset/ionchannels_membraneproteins_imbalanced_test.csv")


def get_embeddings(model, tokenizer, protein_sequences, batch_size=4):
    # Placeholder list to store embeddings
    all_embeddings = []

    # Create batches
    num_batches = len(protein_sequences) // batch_size + \
        (len(protein_sequences) % batch_size != 0)
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size

        batch_sequences = [list(seq)
                           for seq in protein_sequences[start_idx:end_idx]]
        outputs = tokenizer.batch_encode_plus(batch_sequences,
                                              add_special_tokens=False,
                                              padding=True,
                                              truncation=True,
                                              max_length=1024,
                                              is_split_into_words=True,
                                              return_tensors="pt")

        device = get_accelerator().current_device_name()
        outputs = {k: v.to(device) for k, v in outputs.items()}

        # For T5, use the same input_ids as decoder_input_ids.
        outputs["decoder_input_ids"] = outputs["input_ids"]

        with torch.no_grad():
            model_outputs = model(**outputs)
            embeddings = model_outputs.last_hidden_state

            # Max pooling
            mask_expanded = outputs['attention_mask'].unsqueeze(
                -1).expand(embeddings.size()).float()
            embeddings_max = torch.max(embeddings * mask_expanded, 1).values

            all_embeddings.append(
                embeddings_max.clone().cpu().detach())  # deep copy

        # Logging progress to wandb
        wandb.log({
            "Current Batch": batch_num + 1,
            "Total Batches": num_batches,
            "Percentage Completed": (batch_num + 1) / num_batches * 100
        })

    # Concatenate all the embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings


# Getting the embeddings
train_embeddings_toot = get_embeddings(
    toot_plm_p2s_model, tokenizer, train_dataset['sequence'].values)
train_embeddings_ankh = get_embeddings(
    ankh_large_model, tokenizer, train_dataset['sequence'].values)

test_embeddings_toot = get_embeddings(
    toot_plm_p2s_model, tokenizer, test_dataset['sequence'].values)
test_embeddings_ankh = get_embeddings(
    ankh_large_model, tokenizer, test_dataset['sequence'].values)

# Train logistic regressions
lr_toot = LogisticRegression(max_iter=1000).fit(
    train_embeddings_toot, train_dataset['label'])
lr_ankh = LogisticRegression(max_iter=1000).fit(
    train_embeddings_ankh, train_dataset['label'])

# Predictions
preds_toot = lr_toot.predict(test_embeddings_toot)
preds_ankh = lr_ankh.predict(test_embeddings_ankh)

# plot classifier
wandb.sklearn.plot_classifier(lr_toot, train_embeddings_toot,
                              train_dataset['label'], test_embeddings_toot, test_dataset['label'], [
                                  "ion channel", "other membrane protein"], model_name="LogisticRegression",
                              feature_names=None, target_names=None, title="Logistic Regression for TooT-PLM-P2S")

wandb.sklearn.plot_classifier(lr_ankh, train_embeddings_ankh,
                              train_dataset['label'], test_embeddings_ankh, test_dataset['label'], [
                                  "ion channel", "other membrane protein"], model_name="LogisticRegression",
                              feature_names=None, target_names=None, title="Logistic Regression for ankh-large")

# plot roc curve
wandb.sklearn.plot_roc(test_dataset['label'], [
    preds_toot, preds_ankh], ["TooT-PLM-P2S", "ankh-large"], title="ROC Curve for TooT-PLM-P2S and ankh-large")


# Plot confusion matrix
wandb.sklearn.plot_confusion_matrix(test_dataset['label'], preds_toot, [
                                    "ion channel", "other membrane protein"], title="Confusion Matrix for TooT-PLM-P2S")
wandb.sklearn.plot_confusion_matrix(test_dataset['label'], preds_ankh, [
                                    "ion channel", "other membrane protein"], title="Confusion Matrix for ankh-large")

# plot summary metrics
wandb.sklearn.plot_summary_metrics(lr_toot, train_embeddings_toot,
                                   train_dataset['label'], test_embeddings_toot, test_dataset['label'])
wandb.sklearn.plot_summary_metrics(lr_ankh, train_embeddings_ankh,
                                   train_dataset['label'], test_embeddings_ankh, test_dataset['label'])

# Evaluation
accuracy_toot = accuracy_score(test_dataset['label'], preds_toot)
accuracy_ankh = accuracy_score(test_dataset['label'], preds_ankh)


f1_toot = f1_score(test_dataset['label'], preds_toot, average='macro')
f1_ankh = f1_score(test_dataset['label'], preds_ankh, average='macro')


mcc_toot = matthews_corrcoef(test_dataset['label'], preds_toot)
mcc_ankh = matthews_corrcoef(test_dataset['label'], preds_ankh)


wandb.log({
    "accuracy_toot": accuracy_toot,
    "accuracy_ankh": accuracy_ankh,
    "f1_toot": f1_toot,
    "f1_ankh": f1_ankh,
    "mcc_toot": mcc_toot,
    "mcc_ankh": mcc_ankh
})


print(f"Accuracy for toot_plm_p2s: {accuracy_toot}")
print(f"Accuracy for ankh_large: {accuracy_ankh}")
print(f"F1 score for toot_plm_p2s: {f1_toot}")
print(f"F1 score for ankh_large: {f1_ankh}")
print(f"MCC for toot_plm_p2s: {mcc_toot}")
print(f"MCC for ankh_large: {mcc_ankh}")

wandb.finish()
