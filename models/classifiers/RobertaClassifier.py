import torch
from transformers import AutoModel, AutoTokenizer

from models.classifiers import Classifier


class RobertaClassifier(Classifier, torch.nn.Module):
    @classmethod
    def from_pretrained(cls, model_id: str):
        available_ids = ("untrained")

        match model_id:
            case "untrained":
                return cls()  # TODO: implement
            case _:  # Match any pattern
                print(f"No such model id as {model_id}. Available ids are: {available_ids}")
                return ValueError

    def __init__(self, config={}):
        super().__init__(config)

        self.base_model = AutoModel.from_pretrained("roberta-base")  #, self.huggingface_cache)
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")  # TODO: huggingface_cache?
        num_classes = 2
        model_final_layer_dim = 768
        self.prediction_head = torch.nn.Linear(model_final_layer_dim, num_classes)

    def forward(self,input_ids,attention_mask=None,token_type_ids=None) -> torch.Tensor:
        x = self.base_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = self.prediction_head(x["pooler_output"])
        return x

    def predict(self, strings: list[str]) -> dict:
        "Run the model on a list of strings and return results as dict."

        with torch.no_grad():
            tokenized = self.tokenizer(strings)
            model_out: torch.Tensor = self(tokenized)
            confidences = model_out.cpu().numpy()
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

        return ["real", "fake"]


def test():
    print("Testing …")

    clf: RobertaClassifier = RobertaClassifier.from_pretrained("untrained") # type: ignore
    strings = [
        "An exceedingly, almost impossibly long string.",
        "Short"
    ]
    results = clf.predict(strings)
    print("Results:")
    print(results)

if __name__ == "__main__":
    test()