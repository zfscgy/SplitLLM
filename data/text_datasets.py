import os

import numpy as np
from datasets import load_dataset_builder, load_dataset, load_from_disk


storage_dir = "/home/zf/pycharm/SplitLLM/data/storage"


def get_squad():
    # print("Load SQUAD dataset...")
    # squad_builder = load_dataset_builder("squad", cache_dir=storage_dir)
    # print(squad_builder.info.description)
    try:
        squad_datasets = load_from_disk(storage_dir + "/squad")
    except Exception as e:
        print("Dataset not in local disk, downloading...", e)
        squad_datasets = load_dataset("squad")
        squad_datasets.save_to_disk(storage_dir + "/squad")
    return squad_datasets


def get_wikitext():
    try:
        wikitext_datasets = load_from_disk(storage_dir + "/wikitext-2-v1")
    except Exception as e:
        print("Dataset not in local disk, downloading...", e)
        wikitext_datasets = load_dataset("wikitext", "wikitext-2-v1")
        wikitext_datasets.save_to_disk(storage_dir + "/wikitext-2-v1")
    return wikitext_datasets


def get_find_sum():
    try:
        find_sum_datasets = load_from_disk(storage_dir + "/find-sum")
    except Exception as e:
        print("Dataset not in local disk, downloading...", e)
        find_sum_datasets = load_dataset("Sakshi1307/FindSUM")
        find_sum_datasets.save_to_disk(storage_dir + "/find-sum")
    return find_sum_datasets


class SquadDataset:
    def __init__(self):
        self.squad_datasets = get_squad()

    def get_random_sentence_in_context(self, split: str="train"):
        current_dataset = self.squad_datasets[split]
        idx = int(np.random.randint(0, len(current_dataset)))
        context_senteces = current_dataset[idx]['context'].split(".")
        n_senteces = len(context_senteces)
        idx = int(np.random.randint(0, n_senteces))
        return context_senteces[idx].strip()


    def get_question_text(self, index: int, split: str="train"):
        current_dataset = self.squad_datasets[split]
        return current_dataset[index]['question']
    
    def get_question_answer(self, index: int, split: str="train"):
        current_data = self.squad_datasets[split][index]
        return current_data['context'] + current_data['question'], current_data['answers']['text'][0]


class WikitextDataset:
    def __init__(self):
        self.wikitext_datasets = get_wikitext()

    def get_corpus(self, index, split: str="train"):
        current_dataset = self.wikitext_datasets[split]
        return current_dataset[index]['text']


class FindSumDataset:
    def __init__(self):
        self.find_sum_datasets = get_find_sum()

    def get_document(self, index: int, split: str="train"):
        current_dataset = self.find_sum_datasets[split]
        return current_dataset[index]['document']

if __name__ == "__main__":
    os.environ['https_proxy'] = "10.181.173.91:10811"
    find_sum_datasets = get_find_sum()
    print(find_sum_datasets['train']['document'][2224])
