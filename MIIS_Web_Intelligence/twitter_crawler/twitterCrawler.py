import tweepy
import simplejson as json
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from textblob import TextBlob
from tweepy import Stream
import indicoio
from prettytable import PrettyTable
import matplotlib.pyplot as plt


#https://marcobonzanini.com/2015/06/16/mining-twitter-data-with-python-and-js-part-7-geolocation-and-interactive-maps/
#https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/
#https://indico.io/docs/emotion
consumer_key = 'TSGLobOrXce2tGMP7JDLlXzmk'
consumer_secret = 'PYL7ekWDyaUPzluMjAgnbxRQlf5QSXbG2nvJQCzhCbpcRvy7cz'
access_token = '486780877-tRcgcIjSmrDZZ73QoLspfPYOB8cJEp1FMiRamEqj'
access_secret = 'HAWROwUAnzQskrsqZHNjkUZpo7ZU7pt96bvxqiT5GgtWC'
indicoio.config.api_key = 'cf487a6064d4c7d2972c6d8be6953dbd'


# Manage connections
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)


class MyListener (StreamListener):

    def __init__(self, api=None):
        super(StreamListener,self).__init__()
        self.num_tweets = 0

    # Gather tweeters not real time
    def gather_twitter(self):
        for status in tweepy.Cursor(api.home_timeline).items(10):
            print(status.text)
            print()
            print('______')

    def on_data(self,data):
        #'a' --> append to the file every time without delete the file
        #change the file name by a variable (no hardcore)
        try:
            with open('yoga.json','a') as tweet_file:
                tweet_file.write(data)
                self.num_tweets += 1

                return self.num_tweets < 100

        except BaseException as e:
            print("Error occurred is:  %s" %str(e))
        #return True

    def on_error(self,status):
        print("Error: ", status)
        return False

## Analysis sentiment in tweets filtered lang = en
def sentiment_analysis_English():
    fname = 'yoga.json'
    freq = 0
    sum_pol = 0
    sum_subj = 0

    with open(fname, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            if tweet['lang'] == 'en':
                freq += 1
                analysis = TextBlob(line)
                sum_pol += (analysis.sentiment.polarity)
                sum_subj +=(analysis.sentiment.subjectivity)

    return sum_pol,sum_subj,freq

## Analysis sentiment in tweets filtered lang = es
def sentiment_analysis_Spanish():
    fname = 'yoga.json'
    freq = 0
    sum_sent = 0
    with open(fname, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            if tweet['lang'] == 'es':
                freq +=1
                sum_sent += indicoio.sentiment(tweet['text'])

    return sum_sent, freq

def country_analysis():
    fname = 'yoga.json'
    dict = {}
    with open(fname, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            if tweet['place'] is not None:
                if tweet['place']['country'] in dict:
                    dict[tweet['place']['country']] +=1
                else:
                    dict[tweet['place']['country']] = 1
    return dict


# Print table of frequences of tweets
def print_table_freq():
    table = PrettyTable()
    table.title="Frequency tweet #Yoga"
    table.field_names = ["Freq_En", "Freq_Es"]
    table.add_row([freqEn,freqEs])
    print(table)

# Print table of sentiments of tweets
def print_table_sentiment():
    table = PrettyTable()
    table.title="Sentiment analysis of tweets with #Yoga"
    table.field_names = ["Polarity (EN)", "Subjectivity (EN)", "Likelihood Sentiment (ES)"]
    table.add_row([polarity/freqEn, subjectivity/freqEn,sent/freqEs])
    print(table)


def countries_tweets_visualization():
    # Pie chart visualization
    labels = dict.keys()
    sizes = dict.values()

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.show()

twitter_stream = Stream(auth,MyListener())
twitter_stream.filter(track=['yoga'])
polarity, subjectivity, freqEn = sentiment_analysis_English()
sent, freqEs = sentiment_analysis_Spanish()
print_table_freq()
print_table_sentiment()
dict = country_analysis()
