from datasets import load_dataset
import pandas as pd
import os


# We print all the names of the files and folders under ./datasets. separated by folder and the files inside
def print_dataset_names():
    for root, dirs, files in os.walk("./datasets"):
        print(root)
        print(dirs)
        print(files)

print_dataset_names()

# We will print the name of the columns of each dataset of all csv files under ./datasets separate by folder.
# We will also print the number of rows of each dataset
def print_dataset_columns_and_rows():
    for root, dirs, files in os.walk("./datasets"):
        for file in files:
            if file.endswith(".csv"):
                print(root)
                print(file)
                df = pd.read_csv(os.path.join(root, file))
                print(df.columns)
                # print(len(df))

print_dataset_columns_and_rows()
