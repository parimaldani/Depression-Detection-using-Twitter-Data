import csv

from nltk import word_tokenize
from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import AppAuthHandler
from tweepy import Stream
from textblob import TextBlob

import re
import pandas as pd
import numpy as np

import twitter_credentials
from NLTK_Sentiment_Analysis import classifier, remove_noise


# sad = 'depressed OR sad OR #depressed OR #sad'



# # # # TWITTER CLIENT # # # #
class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.search, q='depressed OR sad OR rage OR lonely OR loneliness OR melancholy OR #depressed OR #sad OR #rage OR #lonely OR #loneliness OR #melancholy',
                            lang='en', since="2020-01-01").items(num_tweets):
            tweets.append(tweet)
        return tweets


class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = AppAuthHandler(twitter_credentials.API_KEY, twitter_credentials.API_KEY_SECRET)
        # auth = OAuthHandler(twitter_credentials.API_KEY, twitter_credentials.API_KEY_SECRET)
        # auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth


class Tweet_Analyzer():
    """
    Analyse and categorise content from twitter
    """

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))

        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

    def tweets_to_dataframe(self, tweets):
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['tweets'])

        df['id'] = np.array([tweet.id for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        return df


if __name__ == '__main__':
    twitter_client = TwitterClient()
    tweet_analyzer = Tweet_Analyzer()

    api = twitter_client.get_twitter_client_api()
    tweets = twitter_client.get_user_timeline_tweets(900)

    df = tweet_analyzer.tweets_to_dataframe(tweets)

    df['sentiment'] = np.array([classifier.classify(dict([token, True] for token in remove_noise(word_tokenize(tweet)))) for tweet in df['tweets']])
    # df['sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in df['tweets']])
    # df['tweets1'] = np.array([tweet_analyzer.clean_tweet(tweet) for tweet in df['tweets']])

    df.to_csv("tweets.csv")
    print(df.head(10))
