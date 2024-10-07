"""
File: imapact_score_ranking.py
Author: Dinesh Ram
Date: April 23, 2024,
Description: File for impact score calculation.
"""
from textblob import TextBlob
import nltk
import spacy
from afinn import Afinn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
from transformers import pipeline

nlt = pipeline("sentiment-analysis")
import matplotlib.pyplot as plt
from tabulate import tabulate


def tbr(verb):
    word_blob = TextBlob(verb)
    if word_blob.sentiment.polarity < 0:
        return word_blob.sentiment.polarity * (-5.0)
    elif word_blob.sentiment.polarity == 0:
        return 1

    return word_blob.sentiment.polarity * 5.0


def afnr(verb):
    afn = Afinn(language='en')
    if afn.score(verb) < 0:
        return afn.score(verb) * (-1)
    elif afn.score(verb) == 0:
        return 1
    return afn.score(verb)


def trs(verb, nlt):
    if nlt(verb)[0]['score'] < 0:
        nlt(verb)[0]['score'] * (-5.0)
    elif nlt(verb)[0]['score'] == 0:
        return 1

    return nlt(verb)[0]['score'] * 5.0


def vdr(verb):
    analyzer = SentimentIntensityAnalyzer()
    if analyzer.polarity_scores(verb)['compound'] < 0:
        return analyzer.polarity_scores(verb)['compound'] * (-5.0)
    elif analyzer.polarity_scores(verb)['compound'] == 0:
        return 1
    return analyzer.polarity_scores(verb)['compound'] * 5.0


def rating_score(verb):
    return (tbr(verb) + vdr(verb) + afnr(verb) + trs(verb, nlt)) / 4


def impact_score(dic):
    impact_score = {key: 0 for key in dic.keys()}

    actor = dic.keys()
    for i in actor:
        sum = 0
        f = dic[i][0]
        for j in dic[i][1]:
            sum += rating_score(j)
        impact_score[i] = sum + f

    return impact_score


def max_actor_imp(dic):
    impact = impact_score(dic)
    max_actor = sorted(impact, key=impact.get, reverse=True)
    return max_actor


def action_score(dic):
    impact_score = {key: 0 for key in dic.keys()}
    actor = dic.keys()
    for i in actor:
        sum = 0
        for j in dic[i][1]:
            sum += rating_score(j)
        impact_score[i] = sum

    return impact_score


def freq_score(dic1):
    freq = {key: dic1[key][0] for key in dic1.keys()}
    return freq


def get_ranking(dic1):
    impactscore_dic = impact_score(dic1)
    sort = sorted(impactscore_dic, key=impactscore_dic.get, reverse=True)

    # Prepare data for tabulation
    table = [[rank + 1, name, impactscore_dic[name]] for rank, name in enumerate(sort)]

    # Print the table
    print(tabulate(table, headers=['Rank', 'Name', 'Score'], tablefmt='pretty'))


def get_graph(dic1):
  freq = freq_score(dic1)
  impactscore = impact_score(dic1)
  actionscore = action_score(dic1)
  sort = list(sorted(impactscore, key= impactscore.get, reverse = True))
  keys = list(sort[:10])

  values1 = list(freq[i] for i in keys)
  values2 = list(impactscore[i] for i in keys)
  values3 = list(actionscore[i] for i in keys)
  bar_width = 0.25
  index = range(len(keys))
  plt.figure(figsize=(20, 6))
  plt.bar(index, values1, bar_width, label='Only Frequency model')
  plt.bar([i + bar_width for i in index], values3, bar_width, label='Only action rating model')
  plt.bar([i + 2 * bar_width for i in index], values2, bar_width, label='Impact score model')

  plt.xlabel('Keys')
  plt.ylabel('Values')
  plt.title('Bar Graph for the Three approaches- (Sorted based on impact score results for top 10 actors)')
  plt.xticks([i + bar_width for i in index], keys)
  plt.legend()
  plt.show()
