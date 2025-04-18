import os
import json
from datasets import Dataset


def load_stereoset(path="../dataset/stereoset"):
    with open(os.path.join(path, "stereoset_subset.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    f.close()

    dict_data = {"id": [], "context": [],
                 "biased_sentence": [],
                 "anti_biased_sentence":[],
                 "unrelated_sentence": [],
                 "bias_type": []}
    for i, instance in enumerate(data):
        dict_data["id"].append(i)
        dict_data["context"].append(instance["context"])
        dict_data["biased_sentence"].append(instance["stereotypical_sentences"])
        dict_data["anti_biased_sentence"].append(instance["anti-stereotypical_sentences"])
        dict_data["unrelated_sentence"].append(instance["unrelated"])
        dict_data["bias_type"].append(instance["bias_type"])

    dataset = Dataset.from_dict(dict_data)
    return dataset
