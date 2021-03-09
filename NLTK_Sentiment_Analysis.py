
# Natural Language ToolKit is a main platform for NLP
from nltk.corpus import twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
# for lemmatization
from nltk.tag import pos_tag

# for noise removal
import re
import string

# defining stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# for count of frequent words
from nltk import FreqDist

import random

# for model building and testing
from nltk import classify
from nltk import NaiveBayesClassifier

from nltk.tokenize import word_tokenize

"""
Step 1: Tokenizing data because language by itself cannot be processed
Tokenization will split a sentence into smaller parts for easier processing
Tokenizing will be done on the basis of spaces and punctuation
punkt model is a pre-trained tokenizer that will be used

Step 2: Normalise data all different forms of a word need to be turned into its canonical form (dictionary form) 
eg: ran, run, running, etc. should be turned into run
Lemmatization will be used for this purpose (accurate but time consuming)
word net is a db for english that will help determine base word

Step 3: Noise removal. Removal of Hyperlinks, Twitter Handles, Punctuations and special characters

Step 4: Preparing data for model. positive and negative sentiments will be tested.
"""


# incorporates lemmatisation and normalisation
def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []

    """
    pos_tag gives a tag to each word from the tweet. the word is classified into classes like NN, VB, VBN,
    VBG etc based on its tense
    """
    lemmatizer = WordNetLemmatizer()

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

# returns most frequesnt words from a list of cleaned tweets
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


# converting tokens into a dictionary
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


# only lemmatization and normalisation
def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


# variables for the json files
# strings() method will print all of the tweets within a dataset as strings
# variables are made to make testing easier
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
# tokenizing
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

# removing noise from positive_tweets and negative_tweets
positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

# all_pos_words = get_all_words(positive_cleaned_tokens_list)

# dictionary form of tokens
positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

# add positive label
positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]

# add negative label
negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]

# create whole dataset
dataset = positive_dataset + negative_dataset

# shuffle dataset
random.shuffle(dataset)

# define testing and training datasets
train_data = dataset[:7000]
test_data = dataset[7000:]

classifier = NaiveBayesClassifier.train(train_data)

# print("Accuracy: ", classify.accuracy(classifier, test_data))

# print(classifier.show_most_informative_features(10))

custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."

custom_tokens = remove_noise(word_tokenize(custom_tweet))

# print(classifier.classify(dict([token, True] for token in custom_tokens)))
# checking the most common words in tweet list
# freq_dist_pos = FreqDist(all_pos_words)
# print(freq_dist_pos.most_common(10))

# cecking difference between tokens and noise removed
# print(positive_tweet_tokens[500])
# print(positive_cleaned_tokens_list[500])


# testing if noise is removed
# print(remove_noise(tweet_tokens,stop_words))

# testing to check whether words have been normalised
# print(lemmatize_sentence(tweet_tokens))

# test tokenization
# print(tweet_tokens)
