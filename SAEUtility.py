import pandas as pd
import numpy as np
import GetOldTweets3 as got
import random as rand
import os
import SAEngine as sae
import TweetCleaning as tc
from datetime import date, timedelta

import matplotlib.pyplot as plt

def assembleDataframe(tweets):
    # Creating list and assemble dataframe of chosen tweet data
    text_tweets = [[tweet.date, tweet.permalink, tweet.text, tweet.username, tweet.favorites, tweet.retweets] for tweet in tweets]
    tweets_df = pd.DataFrame(text_tweets,
                             columns=['Datetime', 'Permalink', 'Text', 'Username', 'Favorites', 'Retweets'])

    tweets_df['Engagements'] = tweets_df['Favorites'] + tweets_df['Retweets']
    return tweets_df

#takes in a column and then sums it up X = df.ColumnName | X = df['ColumnName']
def getAvgScore(X): #pass in column name (Score)
    x = X[X != 0]
    if(len(x)>0):
        return x.to_numpy().sum() / len(x)
    else:
        return -1;


def sortByColumn(df, t):
    return df.sort_values(by = t, ascending = False)


def findPos(tweets_df):
    pos = sortByColumn(tweets_df, ['NB-Sentiment Score']);
    mostPos = pos.iloc[0:1]
    print(mostPos[['Text', 'Favorites', 'Engagements']])

    return mostPos

def findNeg(tweets_df):
    pos = sortByColumn(tweets_df, ['NB-Sentiment Score']);
    mostNeg = pos.iloc[99:100]
    print(mostNeg[['Text', 'Favorites', 'Engagements']])
    return mostNeg



def findTopThreeTweets(tweets_df):
    topThree = sortByColumn(tweets_df, ['Engagements']);
    #print(topThree.iloc[0:3])
    #print(topThree[['Text', 'Favorites', 'Engagements']])
    return topThree.iloc[0:3]

def generateHistogram(tweets_df, text_query):
    # histogram/bar
    print("Printing histogram...")
    plt.xlabel('Sentiment Score')
    plt.ylabel('Number of tweets')
    plt.title("Tweets Vs Naive-Bayes Sentiment Score")
    plt.hist(tweets_df['NB-Sentiment Score'], color='#86bf91', rwidth=0.9)
    #print("About to save histogram...")
    plt.savefig('static/{}/plot.png'.format(str(text_query)))
    print("Histogram successfully saved!")
    plt.close()

def createDirectory(text_query):
    #print("Making new {} directory...".format(str(text_query)))
    #path = os.getcwd()
    #path = path + "/static/{}".format(str(text_query))
    #os.mkdir(path)


    print("Making new {} directory...".format(str(text_query)))
    path = os.path.dirname(os.path.realpath(__file__))
    print(os.path)
    os.chdir(path)
    path = path + "/static/{}".format(str(text_query))
    cwd = os.getcwd()
    files = os.listdir(cwd)  # Get all the files in that directory
    print("Files in %r: %s" % (cwd, files))

    os.mkdir(path)

def generatePieGraph(dfc, text_query):
    positive = dfc[dfc >= .2].count()
    negative = dfc[-.2 >= dfc].count()
    neutral = len(dfc) - positive - negative

    size = [positive, neutral, negative]
    colors = ['#86bf91', '#EFEBE8', '#CD5C5C']
    labels = 'Positive', 'Neutral', 'Negative'

    fig1, ax1 = plt.subplots()
    ax1.pie(size, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')

    plt.savefig('static/{}/pie.png'.format(str(text_query)))
    plt.close()

def sentiment_vs_time(text_query, avgScore):
    today = date.today()
    thirtyDaysPast = today - timedelta(30)
    sixtyDaysPast = today - timedelta(60)
    ninetyDaysPast = today - timedelta(90)
    oneHundredTwentyDaysPast = today - timedelta(120)
    oneHundredFiftyDaysPast = today - timedelta(150)
    oneHundredEightyDaysPast = today - timedelta(180)

    count = 10

    tweetCriteria30 = got.manager.TweetCriteria().setQuerySearch(str(text_query)) \
        .setMaxTweets(int(count)) \
        .setTopTweets(1) \
        .setSince(str(sixtyDaysPast))\
        .setUntil(str(thirtyDaysPast))

    tweetCriteria60 = got.manager.TweetCriteria().setQuerySearch(str(text_query)) \
        .setMaxTweets(int(count)) \
        .setTopTweets(1) \
        .setSince(str(ninetyDaysPast)) \
        .setUntil(str(sixtyDaysPast))

    tweetCriteria90 = got.manager.TweetCriteria().setQuerySearch(str(text_query)) \
        .setMaxTweets(int(count)) \
        .setTopTweets(1) \
        .setSince(str(oneHundredTwentyDaysPast)) \
        .setUntil(str(ninetyDaysPast))

    tweetCriteria120 = got.manager.TweetCriteria().setQuerySearch(str(text_query)) \
        .setMaxTweets(int(count)) \
        .setTopTweets(1) \
        .setSince(str(oneHundredFiftyDaysPast)) \
        .setUntil(str(oneHundredTwentyDaysPast))

    tweetCriteria150 = got.manager.TweetCriteria().setQuerySearch(str(text_query)) \
        .setMaxTweets(int(count)) \
        .setTopTweets(1) \
        .setSince(str(oneHundredEightyDaysPast)) \
        .setUntil(str(oneHundredFiftyDaysPast))

    tweets30 = got.manager.TweetManager.getTweets(tweetCriteria30)
    tweets_df30 = assembleDataframe(tweets30)
    cleaned_tweets_df30 = tweets_df30[['Datetime', 'Text']].copy()
    tc.process_text(cleaned_tweets_df30)
    tc.tokenize_words((cleaned_tweets_df30))
    #print("Printing cleaned_tweets_df30:")
    #print(cleaned_tweets_df30)
    sae.score_by_naive_bayes(cleaned_tweets_df30)
    avgScore30 = getAvgScore(cleaned_tweets_df30['NB-Sentiment Score']) * 100

    tweets60 = got.manager.TweetManager.getTweets(tweetCriteria60)
    tweets_df60 = assembleDataframe(tweets60)
    cleaned_tweets_df60 = tweets_df60[['Datetime', 'Text']].copy()
    tc.process_text(cleaned_tweets_df60)
    tc.tokenize_words((cleaned_tweets_df60))
    sae.score_by_naive_bayes(cleaned_tweets_df60)
    avgScore60 = getAvgScore(cleaned_tweets_df60['NB-Sentiment Score']) * 100

    tweets90 = got.manager.TweetManager.getTweets(tweetCriteria90)
    tweets_df90 = assembleDataframe(tweets90)
    cleaned_tweets_df90 = tweets_df90[['Datetime', 'Text']].copy()
    tc.process_text(cleaned_tweets_df90)
    tc.tokenize_words((cleaned_tweets_df90))
    sae.score_by_naive_bayes(cleaned_tweets_df90)
    avgScore90 = getAvgScore(cleaned_tweets_df90['NB-Sentiment Score']) * 100

    tweets120 = got.manager.TweetManager.getTweets(tweetCriteria120)
    tweets_df120 = assembleDataframe(tweets120)
    cleaned_tweets_df120 = tweets_df120[['Datetime', 'Text']].copy()
    tc.process_text(cleaned_tweets_df120)
    tc.tokenize_words((cleaned_tweets_df120))
    sae.score_by_naive_bayes(cleaned_tweets_df120)
    avgScore120 = getAvgScore(cleaned_tweets_df120['NB-Sentiment Score']) * 100

    tweets150 = got.manager.TweetManager.getTweets(tweetCriteria150)
    tweets_df150 = assembleDataframe(tweets150)
    cleaned_tweets_df150 = tweets_df150[['Datetime', 'Text']].copy()
    tc.process_text(cleaned_tweets_df150)
    tc.tokenize_words((cleaned_tweets_df150))
    sae.score_by_naive_bayes(cleaned_tweets_df150)
    avgScore150 = getAvgScore(cleaned_tweets_df150['NB-Sentiment Score']) * 100

    x = [today, thirtyDaysPast, sixtyDaysPast, ninetyDaysPast, oneHundredTwentyDaysPast, oneHundredFiftyDaysPast]
    y = [avgScore, avgScore30, avgScore60, avgScore90, avgScore120, avgScore150]

    plt.plot(x,y)
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score (-100 to 100)')

    plt.savefig('static/{}/timegraph.png'.format(str(text_query)))
    print("Timegraph successfully saved!")
