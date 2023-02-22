# from sentence_transformers import SentenceTransformer
import pandas as pd
import sys
import torch
from tqdm import tqdm
from transformers import AutoModel,AutoTokenizer

sys.path.append(".")
from utils.utils import get_free_gpu

class TextEmbedder:
    def __init__(self) -> None:
        self.device = get_free_gpu()
        self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large",cache_dir="/data/tide-hackaton/cache/huggingface")
        self.model.to(self.device)

    def embed(self, texts: list[str], batch_size=50) -> torch.Tensor:
        embeddings_batches = []
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(texts), batch_size)):
                texts_batch = texts[batch_start:batch_start + batch_size]
                tokenized_texts = self.tokenizer(texts_batch, return_tensors="pt", padding=True, truncation=True)
                tokenized_texts.to(self.device)

                embeddings_batch = self.model(**tokenized_texts).pooler_output
                embeddings_batch /= embeddings_batch.norm(dim=-1, keepdim=True)
                embeddings_batches.append(embeddings_batch.detach().to(dtype=torch.float16).cpu())

        embeddings = torch.concat(embeddings_batches)
        return embeddings


def main():

    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    df = pd.read_pickle("/filserver/tide-hackaton/datasets/Disinformation-challenge-data/twitter/twitter_combined_df.pickle")
    texts = list(df["content_wo_hashtags"])

    embedder = TextEmbedder()
    embeddings = embedder.embed(texts)

    torch.save(embeddings, "/filserver/tide-hackaton/datasets/Disinformation-challenge-data/twitter/twitter_combined_SimCSE_embeddings.pt")
    embeddings /= embeddings.norm(dim=-1, keepdim=True)
    torch.save(embeddings, "/filserver/tide-hackaton/datasets/Disinformation-challenge-data/twitter/twitter_combined_SimCSE_embeddings_normalized.pt")

    print(embeddings)
    print(embeddings.shape)
    print(embeddings.device)

    # embeddings = model.encode(
    #     texts,
    #     batch_size=1024,
    #     show_progress_bar=True,
    #     convert_to_tensor=True,
    #     normalize_embeddings=True,
    # )

if __name__ == "__main__":
    main()