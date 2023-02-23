# FFI-Flounders: TIDE Hackathon 2023 Log

## Monday
Getting to know the provided dataset.

Made a concatenated dataset of all the tweet datasets.
There were significant amount of duplicates, so removing duplicates shrunk the dataset with 25 %.

Focus on hashtags as a first step. Finding common hashtags and words in the dataset using and looking at the word
Kullback-Leibler divergence

Looking for additional datasets. Considering https://ieee-dataport.org/documents/propaganda-and-fake-news-war-ukraine for propaganda.

Sources:
- https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
- https://www.oecd.org/ukraine-hub/policy-responses/disinformation-and-russia-s-war-of-aggression-against-ukraine-37186bde/
- https://euvsdisinfo.eu/disinformation-cases/#

Looking into parameters to classify disinformation.

From the given data set:
    Guardian : Trustworthy
    NY Times: Trustworthy
    Tweets and reddit: unknown

## Tuesday

Modifying the twitter data provided by the Polish Cyber Command by extracting data on a more readable format and cleaning the dataset by removing stopping words etc. The extracted data is added in seperate columns in the data frame. See the [data folder](data/) for more info.

Create tensor representantions of the dataset using SimCSE/Sentence Transformer, omit hashtags to create a model that is more general and does not simply learn hashtags.

TODO:
Sentiment analysis of twitter posts (pos, neg , neutral): https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
Zero-shot classifier: https://huggingface.co/facebook/bart-large-mnli

Explored K means klustering on user data to get an understanding of how to identify bots.

## Wednesday
Creating SimCSE Embeddings of the dataset and clustered them using physical component analyses to identify clusteres that could indicate
meaningful groups. It appears that certain sentiemnts are clealy gathethered and could indicate an natural distinction for classificaiton of Pro Ukraine sentiments.

Worked on the Web application.

Created a Demo data set that contains about 50 samples of manually verified misinformation from the dataset, and 100 random samples.

Implemented a UkraineSentimentClassifier to detect pro-ukrainian sentiment (or the opposite)

Working on zero-shot classifer for fake news


# Ting å gjøre
- demonstrere at man kan legge til nye hypoteser på sparket
- demonstrere at man kan oppdage ny konto
- dokumentere godt. forklare hver modell: hvor den kommer fra, hva vi har gjort.
- oppsummerende skår (vektet)

## Thursday


## Notes

### Ideas

Time log over amount of disinformation to flag events.

Map to show where disinformation is coming from.

Running analysis on the dataset to find which hashtags are common
Manually choosing hashtags to flag as Pro-Russian and Pro-Ukrainian

Train a sentiment analysis network based on hashtags, but remove from tweet to keep the network from simply learning to look at hastags
and provide a more complex analysis of the sentiment of the content.


### Goal Product

A classifier consisting of several steps:

1) Neural network for Ukraine sentiment analysis based on modified dataset

2) Sematic likeness to known propaganda talking points from this article (https://www.oecd.org/ukraine-hub/policy-responses/disinformation-and-russia-s-war-of-aggression-against-ukraine-37186bde/)

3) NLP Classifier for disinformation

Give a total score based on the steps, while also giving explainability through the different metrics.

A report tab that uses metadata such as dates to display disinformation over time.

A draft diagram displaying a potential architecture:

<img src="media/DisinformationAnalyzerDiagram_draft.png"  width="600">


### Models
BERT, SimCSE, T5, Sentence Transformer for sentence embedding.

HuggingFace, transformer module to run and build on existing models

Model + Classification Head for classification.

SimCSE or Sentence Transformer embedded text to for clustering or semantic likeness

### Data
Proxydata: IMDB, ...

*Datasets:*
 - tide-data (fra POL Cyber command)

 An explaination on how we modified the dataset, and scripts to replicate it can be found [here](data/)

Potential extra:
 - kaggle fake news (https://www.kaggle.com/competitions/fake-news/overview)
 - RussiaUkrainePropaganda dataset (https://ieee-dataport.org/documents/propaganda-and-fake-news-war-ukraine)
 - 3 million Russian troll tweets (https://www.kaggle.com/datasets/fivethirtyeight/russian-troll-tweets)

###  Python-packages
- feedparser
- bs4
- matplotlib
- pytorch
- torchvision
- numpy
- scikit-learn (sklearn)
- transformers
- datasets
- ipykernel
- tqdm
- streamlit = 1.18.1
- plotly
- nltk
- argostranslate

BI nevnte for geolokasjon: spacy, geopy

Offline translation from russian to english
https://skeptric.com/python-offline-translation/

