from sklearn.manifold import TSNE
import torch

embeddings = torch.load("/filserver/tide-hackaton/datasets/Disinformation-challenge-data/twitter/twitter_combined_SimCSE_embeddings_normalized.pt")
# embeddings = embeddings[:100]

dim_reducer = TSNE(n_components=2, learning_rate="auto", metric="cosine", init="pca", perplexity=50, n_jobs=32)
embeddings_dim_reduced = dim_reducer.fit_transform(embeddings.numpy())

torch.save(embeddings_dim_reduced, "/filserver/tide-hackaton/datasets/Disinformation-challenge-data/twitter/twitter_combined_SimCSE_embeddings_normalized_tsne-reduced.pt")