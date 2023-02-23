from __future__ import annotations
import pickle
import sys

import numpy as np
import pandas as pd
from transformers import pipeline

if __name__ == "__main__":
    sys.path.append(".")
from models.classifiers import Classifier
from utils.utils import get_free_gpu

class NarrativeRecognitionClassifier(Classifier):
    """
    This classifier predicts whether texts imply one or more known disinformation narratives.
    The texts and narratives are provided at runtime and can be changed for each run.
    This classifier uses the following model available on huggingface: facebook/bart-large-mnli.
    """

    def __init__(self, config: dict = {}):
        self.config_dict = config

        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=get_free_gpu()
        )

    @classmethod
    def from_pretrained(cls, model_id: str | None) -> NarrativeRecognitionClassifier:
        available_ids = ("common-disinfo-narratives")
        assert model_id is None or model_id in available_ids
        return cls()

    def predict(self, strings: list[str], narratives: list[str]) -> dict:
        "Run the model on a list of strings and return results."

        output = self.classifier(strings, narratives, multi_label=True)
        confidences = np.zeros((len(strings), len(narratives)))
        # The confidences output by this model are not returned in the right order.
        # We have to compensate for this, annoyingly.
        for string_i in range(len(strings)):
            for hyp_i in range(len(narratives)):
                hyp_alt_i = output[string_i]["labels"].index(narratives[hyp_i])  # type: ignore
                conf = output[string_i]["scores"][hyp_alt_i]  # type: ignore

                confidences[string_i, hyp_i] = conf


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

        # This method does not apply to this class.
        raise NotImplementedError


def test():
    print("Testing …")

    # Most of these hypotheses are taken from here: https://www.oecd.org/ukraine-hub/policy-responses/disinformation-and-russia-s-war-of-aggression-against-ukraine-37186bde/
    hypotheses = [
        "Classified documents showing Ukraine was preparing an offensive operation against the Donbas",
        "The massacre of civilians in Bucha, Ukraine, during the first month of the war was staged",
        "The United States is developing bioweapons designed to target ethnic Russians and has a network of bioweapons labs in Eastern Europe",
        "Ukraine threatened Russia with invasion",
        "US paratroopers have landed in Ukraine",
        "Ukraine staged the attack on the hospital in Mariupol on 9 March 2022",
        "European universities are expelling Russian students",
        "Ukraine is training child soldiers",
        "The war in Ukraine is a hoax",
        "Russia was not using cluster munitions during its military operation in Ukraine",
        "NATO has a military base in Odessa",
        "Russia does not target civilian infrastructure in Ukraine",
        "Modern Ukraine was entirely created by communist Russia",
        "Crimea joined Russia legally",
        "Ukrainian forces bombed a kindergarten in Lugansk on Feb. 17, 2022",
        "The United States and the United Kingdom sent outdated and obsolete weapons to Ukraine",
        "Nazism is rampant in Ukrainian politics and society, supported by Ukrainian authorities",
        "Anti-Russian forces staged a coup to overthrow the pro-Russia Ukrainian government in 2014",
        "Russian-speaking residents in Donbas have been subjected to genocide",
        "People in power are pedophiles.",  # This was not from the websource cited above, but we found it to be common.
    ]

    clf = NarrativeRecognitionClassifier()
    strings = [
        "An exceedingly, almost impossibly long string.",
        "Roses are red, violets are blue, some things are false, others are true.",
        "Crimea joined Russia legally.",
        "Crimea was occupied by Russia.",
        "Putin is a man.",
    ]
    results = clf.predict(strings, hypotheses)
    print("Results:")
    for i in range(len(strings)):
        print(results["confs"][i, :], strings[i])
    print(results)

def classify_dataset():
    df = pd.read_pickle("/filserver/tide-hackaton/datasets/Disinformation-challenge-data/twitter/twitter_combined_df.pickle")
    texts = list(df["content"])

    # Most of these hypotheses are taken from here: https://www.oecd.org/ukraine-hub/policy-responses/disinformation-and-russia-s-war-of-aggression-against-ukraine-37186bde/
    hypotheses = [
        "Classified documents showing Ukraine was preparing an offensive operation against the Donbas",
        "The massacre of civilians in Bucha, Ukraine, during the first month of the war was staged",
        "The United States is developing bioweapons designed to target ethnic Russians and has a network of bioweapons labs in Eastern Europe",
        "Ukraine threatened Russia with invasion",
        "US paratroopers have landed in Ukraine",
        "Ukraine staged the attack on the hospital in Mariupol on 9 March 2022",
        "European universities are expelling Russian students",
        "Ukraine is training child soldiers",
        "The war in Ukraine is a hoax",
        "Russia was not using cluster munitions during its military operation in Ukraine",
        "NATO has a military base in Odessa",
        "Russia does not target civilian infrastructure in Ukraine",
        "Modern Ukraine was entirely created by communist Russia",
        "Crimea joined Russia legally",
        "Ukrainian forces bombed a kindergarten in Lugansk on Feb. 17, 2022",
        "The United States and the United Kingdom sent outdated and obsolete weapons to Ukraine",
        "Nazism is rampant in Ukrainian politics and society, supported by Ukrainian authorities",
        "Anti-Russian forces staged a coup to overthrow the pro-Russia Ukrainian government in 2014",
        "Russian-speaking residents in Donbas have been subjected to genocide",
        "People in power are pedophiles.",  # This was not from the websource cited above, but we found it to be common.
    ]

    clf = NarrativeRecognitionClassifier()
    class_scores = clf.predict(texts, hypotheses)

    with open("/filserver/tide-hackaton/datasets/Disinformation-challenge-data/twitter/narrative_match_scores.pickle", "wb") as f:
        pickle.dump(class_scores, f)


if __name__ == "__main__":
    # test()
    classify_dataset()