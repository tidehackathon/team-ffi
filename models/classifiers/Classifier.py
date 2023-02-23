from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class Classifier(ABC):
    def __init__(self, config: dict = {}):
        self.config_dict = config
        # self.huggingface_cache = "/data/tide-hackaton/cache/huggingface"

    @classmethod
    # @abstractmethod
    def from_pretrained(cls, model_id: str) -> Classifier:
        raise NotImplementedError

    @abstractmethod
    def predict(self, strings: list[str]) -> dict:
        """
        Run the model on a list of strings and return results.

        Returns a dict with the following content:
        confs (ndarray[float]]): tuple of the confidences for each class. dim 0 refers to each input string, dim 1 refers to each class.
        class (ndarray[int]): for each input string: the class with the highest confidence.
        maxconf (ndarray[float]): for each input string: the highest class confidence.
        """

        raise NotImplementedError

    @abstractmethod
    def get_class_names(self) -> list[str]:
        """Get a list of the class names."""

        raise NotImplementedError



class DummyClassifier(Classifier):
    def __init__(self, config={}):
        super().__init__(config)

    def predict(self, strings: list[str]) -> dict:
        "Run the model on a list of strings and return results."

        confidences_class_0 = np.array([0.9**len(s) for s in strings])
        confidences = np.zeros((len(strings), 2))
        confidences[:, 0] = confidences_class_0
        confidences[:, 1] = 1 - confidences_class_0

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

        return ["sad", "happy"]


def test():
    print("Testing …")

    clf = DummyClassifier()
    strings = [
        "An exceedingly, almost impossibly long string.",
        "Short"
    ]
    results = clf.predict(strings)
    print("Results:")
    print(results)

if __name__ == "__main__":
    test()