import os
from datasets import load_dataset

"""
Downloads the sdgi dataset to a local data folder
"""

def downlod_data():
    dataset = load_dataset("UNDP/sdgi-corpus", split="train")

    os.makedirs("../data/raw", exist_ok=True)

    dataset.save_to_disk("../data/raw/undp_corpus")

if __name__ == "__main__":
    downlod_data()