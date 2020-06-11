import pandas as pd
from nltk.corpus import stopwords
#apply tokenizer
from nltk.tokenize import RegexpTokenizer

def delete_df(tweets_df, text_query):
    # Delete any df that does not contain searched word in tweet
    tweets_df = tweets_df[tweets_df.Text == text_query]
    return tweets_df

def process_text(cleaned_tweets_df):
    # Makes tweets lowercase
    cleaned_tweets_df['Text'] = cleaned_tweets_df['Text'].str.lower()
    # Removes RTs
    cleaned_tweets_df['Text'] = cleaned_tweets_df['Text'].str.replace('rt', '')
    # Removes mentions of other users (@)
    cleaned_tweets_df['Text'] = cleaned_tweets_df['Text'].replace(r'@\w+', '', regex=True)
    #Removes any mentions of websites
    cleaned_tweets_df['Text'] = cleaned_tweets_df['Text'].replace(r'http\S+', '', regex=True)
    cleaned_tweets_df['Text'] = cleaned_tweets_df['Text'].replace(r'www.[^ ]+', '', regex=True)
    # Removes any numbers
    cleaned_tweets_df['Text'] = cleaned_tweets_df['Text'].replace(r'[0-9]+', '', regex=True)
    # Remove Special Characters
    cleaned_tweets_df['Text'] = cleaned_tweets_df['Text'].replace(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]', '', regex=True)

    # Remove Stopwords
    #cleaned_tweets_df['Text'] = cleaned_tweets_df['Text'].apply(lambda x: remove_stopwords(x))


    # Finds repeated tweets and keeps only first one
    cleaned_tweets_df = cleaned_tweets_df.drop_duplicates(subset='Text', keep='first')

    #Find Empty Data Sets & delete (Due to cleaning of tweets above)
    cleaned_tweets_df.dropna(how='any')

    return cleaned_tweets_df

#IGNORE FOR NOW
# def convert_tuple(cleaned_tweets_df):
#     return tuple(cleaned_tweets_df)

def tokenize_words(cleaned_tweets_df):
    #Tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    cleaned_tweets_df['Text'] = cleaned_tweets_df['Text'].apply(lambda x: tokenizer.tokenize(x))

    return cleaned_tweets_df