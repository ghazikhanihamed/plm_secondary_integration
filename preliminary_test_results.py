import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import wandb
import json
import pandas as pd

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
    "./dataset/ionchannels_membraneproteins_imbalanced_train.csv")
test_dataset = pd.read_csv(
    "./dataset/ionchannels_membraneproteins_imbalanced_test.csv")


# Load the saved embeddings
train_embeddings_toot = torch.load('train_embeddings_toot.pt')
train_embeddings_ankh = torch.load('train_embeddings_ankh.pt')
test_embeddings_toot = torch.load('test_embeddings_toot.pt')
test_embeddings_ankh = torch.load('test_embeddings_ankh.pt')


# Train logistic regressions
lr_toot = LogisticRegression(random_state=1).fit(
    train_embeddings_toot, train_dataset['label'])
lr_ankh = LogisticRegression(random_state=1).fit(
    train_embeddings_ankh, train_dataset['label'])

# Predictions
preds_toot = lr_toot.predict(test_embeddings_toot)
preds_ankh = lr_ankh.predict(test_embeddings_ankh)

# # plot classifier
# wandb.sklearn.plot_classifier(lr_toot, train_embeddings_toot,
#                               train_dataset['label'], test_embeddings_toot, test_dataset['label'], [
#                                   "ion channel", "other membrane protein"], model_name="LogisticRegression",
#                               feature_names=None)

# wandb.sklearn.plot_classifier(lr_ankh, train_embeddings_ankh,
#                               train_dataset['label'], test_embeddings_ankh, test_dataset['label'], [
#                                   "ion channel", "other membrane protein"], model_name="LogisticRegression",
#                               feature_names=None)

# # plot roc curve
# wandb.sklearn.plot_roc(test_dataset['label'], [
#     preds_toot, preds_ankh], ["TooT-PLM-P2S", "ankh-large"])


# # Plot confusion matrix
# wandb.sklearn.plot_confusion_matrix(test_dataset['label'], preds_toot, [
#                                     "ion channel", "other membrane protein"])
# wandb.sklearn.plot_confusion_matrix(test_dataset['label'], preds_ankh, [
#                                     "ion channel", "other membrane protein"])

# # plot summary metrics
# wandb.sklearn.plot_summary_metrics(lr_toot, train_embeddings_toot,
#                                    train_dataset['label'], test_embeddings_toot, test_dataset['label'])
# wandb.sklearn.plot_summary_metrics(lr_ankh, train_embeddings_ankh,
#                                    train_dataset['label'], test_embeddings_ankh, test_dataset['label'])

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
