import nltk.data
import pandas as pd
import numpy as np
import os as os

from textblob.classifiers import NaiveBayesClassifier

def train():

    path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(path)

    df_from_csv = pd.read_csv('PosNegTrainer.csv')

    subset = df_from_csv[['Text', 'Sent']]
    training_set_from_csv = [tuple(x) for x in subset.to_numpy()]

    #print(training_set_from_csv)

    return NaiveBayesClassifier(training_set_from_csv)

