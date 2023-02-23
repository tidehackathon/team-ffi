from __future__ import annotations

import numpy as np


# def sigmoid(x: float):
#         return 1.0 / (1.0 + np.exp(-x))

class ClassifierSummarizer():
    def __init__(self):
        self.clf_weights = {
            "RusUkrWarRelevance": 1,
            "narrativeRecognition": 5,
            "misInformationRoberta": 3,
            "misInformationSimCSE": 3,
            "botDetector": 3,
        }
        # self.certainty = 2

    def predict(self, classifier_results: dict[str, float]) -> float:
        """
        Aggregrate the results of all classifiers into a single score.
        classifier_results is a dict on the following format:
        {
            "classifiername": score,
        }
        """

        total_score = 0
        total_weight = 0
        for clf_name, clf_score in classifier_results.items():
            if clf_name not in self.clf_weights:
                print(f"Warning: ClassifierSummarizer has no weight for the classifier {clf_name}! Will assign it zero weight.")
            else:
                total_score += self.clf_weights[clf_name] * clf_score
                total_weight += self.clf_weights[clf_name]

        weighted_average_score = total_score / total_weight
        # logits = sigmoid_factor * total
        # score = sigmoid(logits)

        return weighted_average_score


def test():
    print("Testing …")

    clf = ClassifierSummarizer()
    classifier_results = {
        "RusUkrWarRelevance": 0.46,
        "narrativeRecognition": 0.005,
        "misInformationRoberta": 0.43,
        "misInformationSimCSE": 0.19,
        "botDetector": 0.5,
    }

    results = clf.predict(classifier_results)
    print("Results:")
    print(results)

if __name__ == "__main__":
    test()