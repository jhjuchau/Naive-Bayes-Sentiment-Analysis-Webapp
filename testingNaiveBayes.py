import nltk.data
import pandas as pd
import numpy as np

from textblob.classifiers import NaiveBayesClassifier

train = [
            ('I love this sandwich.', 'positive'),
            ('this is an amazing place!', 'positive'),
            ('beer', 'positive'),
            ('this is my best work.', 'positive'),
            ("what an awesome view", 'positive'),
            ("he was sworn into office today! god i hate him!", 'negative'),
            ("he was sworn into office today! im so excited!", 'positive'),
            ('I do not like this restaurant', 'negative'),
            ('i am tired of this stuff.', 'negative'),
            ("i can't deal with this", 'negative'),
            ('he is my sworn enemy!', 'negative'),
            ('my boss is horrible.', 'negative'),
            ('I dont care one way or the other', 'neutral'),
            ('idk', 'neutral'),
            ('idc', 'neutral')
        ]

df = pd.read_csv('os.path.dirname(os.path.realpath(__file__))')

subset = df[['Text', 'Sent']]
training_set = [tuple(x) for x in subset.to_numpy()]

#print(training_set)

cl = NaiveBayesClassifier(training_set)

dist = cl.prob_classify("he was sworn into office today")

print(dist.prob("Positive"))
print(dist.prob("Negative"))
print(dist.prob("Neutral"))



