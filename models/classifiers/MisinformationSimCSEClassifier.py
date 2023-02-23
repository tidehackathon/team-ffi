from __future__ import annotations
import pickle
import sys

import numpy as np
import torch

if __name__ == "__main__":
    sys.path.append(".")

from models.classifiers import Classifier
from models.language_model import TextEmbedder

import torch
import tqdm
from transformers import AutoModel,AutoTokenizer
from utils.utils import get_free_gpu

class MisinformationSimCSEClassifier(Classifier):
    """

    """
    # TODO: beskrivelse

    def __init__(self, config: dict = {}):
        self.config_dict = config

        self.text_embedder = TextEmbedder()
        with open("/filserver/tide-hackaton/models/misinf_simcse_linear_logregobj.pickle", "rb") as f:
            self.logreg_clf = pickle.load(f)

    @classmethod
    def from_pretrained(cls, model_id: str | None) -> MisinformationSimCSEClassifier:
        available_ids = ("0")
        assert model_id is None or model_id in available_ids
        return cls()


    def predict(self, strings: list[str]) -> dict:
        "Run the model on a list of strings and return results."

        # Run language model to create text embeddings
        embeddings = self.text_embedder.embed(strings)
        # Run logistic regression model
        confidences = self.logreg_clf.predict_proba(embeddings)

        max_confidence = confidences.max(axis=1)
        max_class = confidences.argmax(axis=1)

        result_dict = {
            "confs": confidences,
            "maxconf": max_confidence[0],
            "class": max_class,
        }
        return result_dict

    def get_class_names(self) -> list[str]:
        """Get a list of the class names."""

        return ["Normal information", "Misinformation" ]


def test():
    print("Testing …")

    clf = MisinformationSimCSEClassifier()
    strings = [
        "An exceedingly, almost impossibly long string.",
        "Roses are red, violets are blue, some things are false, others are true.",
        "This is not a war, but a special military operation.",
        "The president of Russia will give a speech next friday.",
        "Pray for Ukraine!",
        "Ukraine needs our help in this difficult situation.",
        "Ukraine has been invaded by Russia.",
        "The US is planning an invasion on Russia.",
        "The US is planning an invasion on Russia, and Ukraine is acting as a puppet of Biden.",
        "Norway and China are discussing the fish trade.",
        "2022-02-24 @thomas",
        "There are nazi biolabs in Ukraine",
        "There are no nazi biolabs in Ukraine"
    ]
    results = clf.predict(strings)
    print("Results:")
    for i in range(len(strings)):
        print(results["confs"][i, 1], strings[i])
    print(results)

if __name__ == "__main__":
    test()