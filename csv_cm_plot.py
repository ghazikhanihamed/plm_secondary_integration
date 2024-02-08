import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

path = "./confusion_matrices/"


def save_cm(csv_ankh, csv_p2s, title):
    # Load the CSV files into DataFrames
    df_ankh = pd.read_csv(os.path.join(path, f"{csv_ankh}"), header=None)
    df_toot_plm_p2s = pd.read_csv(os.path.join(path, f"{csv_p2s}"), header=None)

    # Set up the matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the first confusion matrix
    sns.heatmap(df_ankh, annot=True, fmt="g", ax=axes[0], cmap="Blues")
    axes[0].set_title("Ankh Model Confusion Matrix")
    axes[0].set_xlabel("Predicted labels")
    axes[0].set_ylabel("True labels")

    # Plot the second confusion matrix
    sns.heatmap(df_toot_plm_p2s, annot=True, fmt="g", ax=axes[1], cmap="Greens")
    axes[1].set_title("TooT-PLM-P2S Model Confusion Matrix")
    axes[1].set_xlabel("Predicted labels")
    axes[1].set_ylabel("True labels")

    plt.tight_layout()
    # plt.show()
    plt.savefig(title, dpi=300, bbox_inches="tight")
    plt.close()


# we make a for-loop to save all the confusion matrices with the filename as cm_[task]_model.csv
tasks = ["ionchannels", "localization", "mp", "transporters", "solubility"]
models = ["ankh", "p2s"]

for task in tasks:
        save_cm(
            f"cm_{task}_ankh.csv",
            f"cm_{task}_p2s.csv",
            f"./plots/cm_{task}.png",
        )
