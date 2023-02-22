from __future__ import annotations
import pickle
import sys

import numpy as np
import torch

if __name__ == "__main__":
    sys.path.append(".")
from models.classifiers import Classifier
from models.language_model import TextEmbedder

class SimpleUkraineSupportClassifier(Classifier):
    """
    This is a hand-crafted linear classifier created by doing 2-dimensionalpricipal component analysis (PCA)
    of SimCSE embeddings of tweets with their hashtags removed and identifying that
     tweets that originally had the hashtag #StandWithUkraine, are relatively well separated from the rest.
    The line was selected manually from within the 2 dimensions chosen by PCA.
    """

    def __init__(self, config: dict = {}):
        self.config_dict = config

        self.text_embedder = TextEmbedder()

        with open("/filserver/tide-hackaton/models/pca_2d_tweets.pickle", "rb") as f:
            self.pca_dim_reducer = pickle.load(f)


    @classmethod
    def from_pretrained(cls, model_id: str | None) -> SimpleUkraineSupportClassifier:
        available_ids = ("handcrafted")
        assert model_id is None or model_id in available_ids
        return cls()

    def _sigmoid(self, x: np.ndarray):
        return 1.0 / (1.0 + np.exp(-x))

    def _linear_clf(self, embeddings_pca):
        border_a = 1.6
        border_b = -0.27
        sigmoid_factor = 2

        distance_from_line = (border_a*embeddings_pca[:, 0] - embeddings_pca[:, 1] + border_b) / (border_a**2 + 1)**0.5
        # scores = distance_from_line
        scores = self._sigmoid(sigmoid_factor * distance_from_line)
        return scores

    def predict(self, strings: list[str]) -> dict:
        "Run the model on a list of strings and return results."

        # Run language model to create text embeddings
        embeddings = self.text_embedder.embed(strings)
        # Reduce text embeddings with PCA
        embeddings_pca = self.pca_dim_reducer.transform(embeddings)

        # Determine signed distance from hand-crafted line
        confidences_class_1 = self._linear_clf(embeddings_pca)

        confidences = np.zeros((len(strings), 2))
        confidences[:, 0] = 1 - confidences_class_1
        confidences[:, 1] = confidences_class_1

        max_confidence = confidences.max(axis=1)
        max_class = confidences.argmax(axis=1)

        result_dict = {
            "confs": confidences,
            "maxconf": max_confidence,
            "class": max_class,
        }
        return result_dict

    def get_class_names(self) -> list[str]:
        """Get a list of the class names."""

        return ["Ukraine supportive", "Not Ukraine supportive", ]


def test():
    print("Testing …")

    clf = SimpleUkraineSupportClassifier()
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
        "2022-02-24 @thomas"
    ]
    results = clf.predict(strings)
    print("Results:")
    for i in range(len(strings)):
        print(results["confs"][i, 0], strings[i])
    print(results)

if __name__ == "__main__":
    test()