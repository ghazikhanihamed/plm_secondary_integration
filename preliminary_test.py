from transformers import AutoModel, AutoTokenizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from accelerate import Accelerator
import torch

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

toot_plm_p2s_model_name = "ghazikhanihamed/TooT-PLM-P2S"
ankh_large_model_name = "ElnaggarLab/ankh-large"

toot_plm_p2s_model = AutoModel.from_pretrained(toot_plm_p2s_model_name)
ankh_large_model = AutoModel.from_pretrained(ankh_large_model_name)

toot_plm_p2s_model.to(device)
ankh_large_model.to(device)

tokenizer = AutoTokenizer.from_pretrained(ankh_large_model_name)

train_dataset = pd.read_csv(
    "./dataset/ionchannels_membraneproteins_imbalanced_train.csv")
test_dataset = pd.read_csv(
    "./dataset/ionchannels_membraneproteins_imbalanced_test.csv")


def get_embeddings(model, tokenizer, protein_sequences, batch_size=32):
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
                                              add_special_tokens=True,
                                              padding=True,
                                              is_split_into_words=True,
                                              return_tensors="pt")
        outputs = {key: value.to(device) for key, value in outputs.items()}
        with torch.no_grad():
            model_outputs = model(**outputs)
            if isinstance(model_outputs, tuple):
                embeddings = model_outputs[0]
            else:
                embeddings = model_outputs

            # Max pooling
            mask_expanded = outputs['attention_mask'].unsqueeze(
                -1).expand(embeddings.size()).float()
            embeddings_max = torch.max(embeddings * mask_expanded, 1).values
            all_embeddings.append(embeddings_max.cpu())

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

# Evaluation
accuracy_toot = accuracy_score(test_dataset['label'], preds_toot)
accuracy_ankh = accuracy_score(test_dataset['label'], preds_ankh)

f1_toot = f1_score(test_dataset['label'], preds_toot, average='macro')
f1_ankh = f1_score(test_dataset['label'], preds_ankh, average='macro')

mcc_toot = matthews_corrcoef(test_dataset['label'], preds_toot)
mcc_ankh = matthews_corrcoef(test_dataset['label'], preds_ankh)

print(f"Accuracy for toot_plm_p2s: {accuracy_toot}")
print(f"Accuracy for ankh_large: {accuracy_ankh}")
print(f"F1 score for toot_plm_p2s: {f1_toot}")
print(f"F1 score for ankh_large: {f1_ankh}")
print(f"MCC for toot_plm_p2s: {mcc_toot}")
print(f"MCC for ankh_large: {mcc_ankh}")
