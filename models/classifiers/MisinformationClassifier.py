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

class BinaryClassifier(torch.nn.Module):
    def __init__(self,model,model_dim,out_classes):
        super().__init__()
        self.model = model
        self.predict = torch.nn.Linear(model_dim,out_classes)
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self,input_ids,attention_mask=None,token_type_ids=None):
        x=self.model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)

        y=self.predict(self.dropout(x["pooler_output"])) 
        return y

class MisinformationClassifier(Classifier):
    """
    This is a finetuned LLM (roberta-base) designed to classify misinformation. The misinformation data was gathered from 
    the twitter dataset by using a pretrained zero-shot classifier (bart-large-mnli) to find tweets that contain known disinformation
    narratives about the Russian invasion of Ukraine.

    roberta-base: https://huggingface.co/roberta-base
    bart-large-mnli: https://huggingface.co/facebook/bart-large-mnli

    """

    def __init__(self, config: dict = {}):
        self.config_dict = config
        self.device = get_free_gpu()
        model_dim=768
        classes=2

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        base_model = AutoModel.from_pretrained("roberta-base")
        self.model = BinaryClassifier(base_model,model_dim,classes)

        self.model.load_state_dict(torch.load("/filserver/tide-hackaton/datasets/Disinformation-challenge-data/twitter/misinf_statedict.pt"))
        self.model.to(self.device)
        self.model.eval()


    @classmethod
    def from_pretrained(cls, model_id: str | None) -> MisinformationClassifier:
        available_ids = ("handcrafted")
        assert model_id is None or model_id in available_ids
        return cls()


    def predict(self, strings: list[str]) -> dict:
        "Run the model on a list of strings and return results."

        with torch.no_grad():
            # tokenize inputs
            tokenized_inputs = self.tokenizer(strings,return_tensors="pt",padding=True,truncation=True)
            ids=tokenized_inputs.input_ids.to(self.device)
            attention_mask=tokenized_inputs.attention_mask.to(self.device)

            # predict on tokens
            logits = self.model(ids,attention_mask=attention_mask)
            confidences = torch.nn.functional.softmax(logits)

            max_confidence = confidences.max(axis=1)
            max_class = confidences.argmax(axis=1)

            result_dict = {
                "confs": confidences.cpu().numpy(),
                "maxconf": max_confidence[0].cpu().numpy(),
                "class": max_class.cpu().numpy(),
            }
        return result_dict

    def get_class_names(self) -> list[str]:
        """Get a list of the class names."""

        return ["Normal information", "Misinformation" ]


def test():
    print("Testing …")

    clf = MisinformationClassifier()
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
        print(results["confs"][i, 0], strings[i])
    print(results)

if __name__ == "__main__":
    test()