import twitter
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
import csv
import authenticate
from datetime import datetime, date, timedelta
from dateutil.parser import parse
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
# sudo nano ~/anaconda2/envs/twitter/lib/python2.7/site-packages/vaderSentiment/vaderSentiment.py -> add from io import open
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

api = authenticate.login

# INPUT NEEDED HERE!
searchwords = ["EnterTwitterUsernameHere_1", "EnterTwitterUsernameHere_2"]  # Enter any number (More than 2 possible) of Twitter Usernames Here, for them to be parsed

# INPUT NEEDED HERE!
csvFile = open('enterFilenameHere.csv', 'a')  # Filename for output .csv can be entered here!

csvWriter = csv.writer(csvFile)

bag = ['good', 'bad', 'gross', 'nasty', 'tasty', 'delicious', 'dirty',
       'disgusting', 'lame', 'cheap', 'hate', 'like', 'love', 'vile', 'nice',
       'satisfying', 'enjoy', 'enjoyable', 'satisfy', 'tempting', 'comfort',
       'comforting', 'happy', 'glad', 'joy', 'yucky', 'yuck', 'flat', 'stale',
       'boring', 'depressing', 'sad', 'unhappy', 'annoy', 'annoying', 'anger',
       'angry', 'outrage', 'outragous', 'enrage', 'enraging', 'unpleasant',
       'flavor', 'flavorful', 'yum', 'yummy', 'mouthwatering', 'delectable',
       'pleasant', 'pleasing', 'exquisite', 'fire', 'offensive', 'repulsive',
       'sick', 'sickening', 'awful', 'detestable', 'flavorless', 'tasteless',
       'savory', 'unsavory', 'ok', 'fine', 'decent', 'alright', 'adequate',
       'well', 'ill', 'poor', 'delight', 'evil', 'misery', 'vegan', 'burger',
       'taco', 'people', 'eating', 'food', 'restaurant', 'fries', 'vegans',
       'chicken', 'beef', 'pork', 'fish', 'eat', 'go', 'since',
       'nuggets', 'trending', 'coffee', 'coke', 'pepsi', 'sprite', 'large',
       'medium', 'small', 'sweet', 'kids', 'sauce', 'pie', 'pies', 'delivery',
       'bag', 'cheese', 'friend', 'fresh', 'frozen', 'burrito', 'steak',
       'more', 'ubereats', 'order', 'bring', 'please', 'want', 'need', 'menu',
       'location', 'try', 'tried', 'drive', 'spicy', 'bacon', 'time', 'store',
       'meal', 'thanks', 'cream', 'shake', 'onion', 'sandwich', 'chocolate',
       'coupon', 'ice', 'service', 'meat', 'better', 'excited', 'great',
       'deal', 'buy', 'year', 'poisining', 'night', 'poisened', 'really', 'rt',
       '@ubereats', 'have', 'what', 'why', 'that', 'how', 'we',
       'got', 'about', 'make', 'from', 'guys', 'should', 'some', 'app', 'best',
       'only', 'never', 'double', 'free', 'employees', 'right',
       'stop', 'ever', 'today', 'minutes', 'because', "the",
       "be", "of", "and", "a", "to", "in", "he", "it", "for", "they", "I",
       "with", "as", "not", "on", "she", "at", "by", "this", "you", "do", "but",
       "or", "which", "one", "would", "all", "will", "there", "say", "who",
       "when", "can", "if", "no", "man", "out", "other", "so",
       "up", "than", "into", "could", "state", "new",
       "take", "come", "these", "know", "see", "use", "get",
       "then", "first", "any", "work", "now", "may", "such", "give", "over", "think",
       "most", "even", "find", "day", "also", "after", "way", "many", "must", "look",
       "before", "back", "through", "long", "where", "much",
       "down", "own", "just", "each", "those", "feel",
       "seem", "high", "too", "place", "little", "world", "very", "still",
       "nation", "hand", "old", "life", "tell", "write", "become", "here", "show",
       "house", "both", "between", "mean", "call", "develop", "under", "last",
       "move", "thing", "general", "school", "same", "another",
       "begin", "while", "number", "part", "turn", "real", "leave", "might",
       "point", "form", "off", "child", "few", "against", "ask",
       "late", "home", "interest", "person", "end", "open", "public",
       "follow", "during", "present", "without", "again", "hold", "govern", "around",
       "possible", "head", "consider", "word", "program", "problem", "however", "lead",
       "system", "set", "eye", "plan", "run", "keep", "face", "fact", "group",
       "play", "stand", "increase", "early", "course", "change", "help", "line"]

vectorizer = CountVectorizer(min_df=1, vocabulary=bag)


def getBinary(data):
    if data is "":
        return 0
    elif data is None:
        return 0
    elif len(data) is 0:
        return 0
    else:
        return 1


def getBool(data):
    if data is True:
        return 1
    else:
        return 0


def getLength(data):
    return len(data)


def getMod(data):
    if data % 2 is 0:
        return 0
    else:
        return 1


def getDate(data):
    dt = parse(data)
    return dt.strftime("%Y%m%d")


def getTime(data):
    dt = parse(data)
    return dt.strftime("%H%M%S")


def getClass(data):
    if data is searchwords[0]:
        return 0
    if data is searchwords[1]:
        return 1
    if data is searchwords[2]:
        return 2
    if data is searchwords[3]:
        return 3


def getCurrDate():
    today = date.today()
    ago = date.today() - timedelta(days=10)
    return today, ago


def cleanTweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def getTweetSentiment(data):
    analysis = TextBlob(cleanTweet(data))
    if analysis.sentiment.polarity > 0:
        return 0  # positive
    elif analysis.sentiment.polarity == 0:
        return 1  # neutral
    else:
        return 2  # negative


def getVaderAnalyze(data):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(cleanTweet(data))
    neg, neu, pos, comp = vs.items()
    neg, neu, pos, comp = neg[1], neu[1], pos[1], comp[1]
    return neg, neu, pos, comp


csvWriter.writerow(['createdatdate', 'createdattime', 'id',
                    'textencode', 'hashtags',
                    'usermentions', 'usercreatedatedate',
                    'usercreatedatetime', 'nameencode',
                    'screennameencode', 'locationencode',
                    'followerscount', 'friendscount',
                    'defaultprofile', 'descriptionencode',
                    'favouritescount', 'verified',
                    'defaultprofileimage', 'timezone',
                    'profilebackgroundcolor',
                    'place', 'retweetcount', 'favoritecount',
                    'possiblysensitive'] + bag + ['blobsentiment',
                                                  'neg', 'neu', 'pos', 'comp', 'class'])

for search in searchwords:
    today, ago = getCurrDate()
    while ago <= today:
        ago = ago + timedelta(days=1)
        tweets = api.GetSearch(" -filter:retweets" + " to:" + search,
                               count=100, lang="en", until=ago)  # + " filter:safe"
        for tweet in tweets:
            currtweet = [tweet.text.encode('utf-8')]
            dtm = vectorizer.fit_transform(currtweet)
            dataframe = pd.DataFrame(
                dtm.toarray(), index=currtweet, columns=vectorizer.get_feature_names())
            dlist = dataframe.values.tolist()
            dlist = dlist[0]
            neg, neu, pos, comp = getVaderAnalyze(tweet.text)
            csvWriter.writerow([getDate(tweet.created_at), getTime(tweet.created_at), getMod(tweet.id),
                                getLength(tweet.text.encode('utf-8')
                                          ), getBinary(tweet.hashtags),
                                getLength(tweet.user_mentions), getDate(
                                    tweet.user.created_at),
                                getTime(tweet.user.created_at), getLength(
                                    tweet.user.name.encode('utf-8')),
                                getLength(tweet.user.screen_name.encode(
                                    'utf-8')), getBinary(tweet.user.location.encode('utf-8')),
                                tweet.user.followers_count, tweet.user.friends_count,
                                getBool(tweet.user.default_profile), getLength(
                                    tweet.user.description.encode('utf-8')),
                                tweet.user.favourites_count, getBool(
                                    tweet.user.verified),
                                getBool(tweet.user.default_profile_image), getBinary(
                                    tweet.user.time_zone),
                                getBinary(tweet.user.profile_background_color),
                                getBinary(
                                    tweet.place), tweet.retweet_count, tweet.favorite_count,
                                getBool(tweet.possibly_sensitive)] + dlist + [getTweetSentiment(tweet.text),
                                                                              neg, neu, pos, comp, search])
