from datasets import load_from_disk

"""
Helper function to remove an item from a dictionary and returns
the same dictionary with the item removed

Parameters
-----------
dict: dict
    The dictionary from which an item should be removed
item: V
    The key to the item which should be removed

Outputs
--------
d: dict
    The dictionary without a particular item
"""
def remove_dict_item(dict, item):
    d = dict
    del d[item]
    return d

def preprocess_data():
    # Drop the embeddings
    dataset = load_from_disk("/home/michal/Documents/sdgi-modelling/data/raw/undp_corpus")
    
    dataset = dataset.map(lambda row: remove_dict_item(row, "embedding"))
    print(dataset[0])


if __name__ == "__main__":
    preprocess_data()
    