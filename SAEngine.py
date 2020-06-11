import pandas as pd
import numpy as np
import random as rand
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
#from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob

from textblob.classifiers import NaiveBayesClassifier
import NaiveBayesTrainer as nbtrainer

#for testing
def random_scores(tweets):
    return np.array([rand.randint(1, 100) for tweet in tweets]);

#---------------------

def sentimentloop(cleaned_tweets):
    score_column = np.empty([100, 1])
    count = 0
    #for x in cleaned_tweets:
    #    score_column[count] = (TextBlob(str(cleaned_tweets['Text'])).sentiment.polarity)
    #    count = count + 1;
    for index, row in cleaned_tweets.iterrows():
        score_column[count] = TextBlob(str(row['Text'])).sentiment.polarity
    return score_column


def analyze_tweets(cleaned_tweets):
    cleaned_tweets['Polarity'] = ""
    cleaned_tweets['Subjectivity'] = ""
    count = 0
    size = len(cleaned_tweets.index)
    while count != size:
        cleaned_tweets['Polarity'][count] = TextBlob(str(cleaned_tweets['Text'][count])).sentiment.polarity
        cleaned_tweets['Subjectivity'][count] = TextBlob(str(cleaned_tweets['Text'][count])).sentiment.subjectivity
        count = count + 1;
    return 0;


def score_by_naive_bayes(cleaned_tweets):
    cleaned_tweets['NB-Positive Probability'] = ""
    cleaned_tweets['NB-Negative Probability'] = ""
    cleaned_tweets['NB-Sentiment Score'] = ""
    cl = nbtrainer.train()
    count = 0
    size = len(cleaned_tweets.index)

    while count != size:
        dist = cl.prob_classify(str(cleaned_tweets['Text'][count]))
        cleaned_tweets['NB-Positive Probability'][count] = dist.prob("Positive")
        cleaned_tweets['NB-Negative Probability'][count] = dist.prob("Negative")
        cleaned_tweets['NB-Sentiment Score'][count] =      dist.prob("Positive") - dist.prob("Negative")
        count = count + 1;
    return 0;

