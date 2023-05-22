from flask import Flask, request, jsonify
import os

os.system("pip install -q pyspark")
os.system("pip install -U -q PyDrive")
os.system("apt-get update -qq")
os.system("apt install openjdk-8-jdk-headless -qq")
os.system("pip install -q graphframes")
os.system("pip install fsspec")
os.system("pip install gcsfs")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
# graphframes_jar = 'https://repos.spark-packages.org/graphframes/graphframes/0.8.2-spark3.2-s_2.12/graphframes-0.8.2-spark3.2-s_2.12.jar'
# spark_jars = '/usr/local/lib/python3.7/dist-packages/pyspark/jars'
# !wget -N -P $spark_jars $graphframes_jar
# os.system("wget -N -P $spark_jars $graphframes_jar")

from flask import Flask, request, jsonify
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from graphframes import *
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from timeit import timeit
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from google.cloud import storage
import os
import math
import numpy as np
import builtins
from numpy import dot
from numpy.linalg import norm
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1

from inverted_index_gcp import *
from inverted_index_gcp import read_posting_list

# building titleWithStem, text and anchor text indexes
os.system('mkdir titleWithStem')
os.system('mkdir dictionaries')
os.system('mkdir bodyWithStem')
os.system('mkdir anchorText')
os.system('mkdir bodyWithoutStem')
os.system('mkdir titleWithoutStem')

os.system('gsutil -m cp gs://title_with_stem/postings_gcp/* "titleWithStem"')  # titleWithStem
os.system('gsutil -m cp gs://omarwa-ir3-bucket1/pr/* "dictionaries"')
os.system('gsutil -m cp gs://body_dictinary_207_206/postings_gcp/* "bodyWithStem"')
os.system('gsutil -m cp gs://anchor_bucket_207_206/postings_gcp/* "anchorText"')
os.system('gsutil -m cp gs://body_dictionary_withoutstem/postings_gcp/* "bodyWithoutStem"')
os.system('gsutil -m cp gs://title_dictionary_withoutstem/postings_gcp/* "titleWithoutStem"')
os.system('gsutil -m cp gs://omarwa-ir3-bucket1/postings_gcp/pageView.pkl "dictionaries"')
os.system('gsutil -m cp gs://body_dictionary_withoutstem/postings_gcp/DL.pkl "dictionaries"')
os.system('gsutil -m cp gs://title_with_stem/postings_gcp/titles.pkl "dictionaries"')
os.system('gsutil -m cp gs://body_dictinary_207_206/postings_gcp/DL.pkl "dictionaries"')

# indexes
title_index_stem = InvertedIndex.read_index("/content/titleWithStem", "index")
title_index = InvertedIndex.read_index("/content/titleWithoutStem", "index")
anchor_index = InvertedIndex.read_index("/content/anchorText", "index")
body_index_stem = InvertedIndex.read_index("/content/bodyWithStem", "index")
body_index = InvertedIndex.read_index("/content/bodyWithoutStem", "index")

# Dictionaries
pageRank = pd.read_csv("/content/dictionaries/part-00000-2641c0bc-346e-4a3e-92eb-a48df309068f-c000.csv.gz",
                       encoding='utf-8', compression='gzip')
a_file = open("/content/dictionaries/titles.pkl", "rb")
titleDict = pickle.load(a_file)
a_file = open("/content/dictionaries/DL.pkl", "rb")
DL = pickle.load(a_file)
a_file = open("/content/dictionaries/pageView.pkl", "rb")
pageView = pickle.load(a_file)

### help functions ###
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stemmer = PorterStemmer()


def tokenize(text, use_stemming=False):

    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    temp = []

    for tok in tokens:
        if tok not in all_stopwords:
            temp.append(tok)
    tokens = temp

    if use_stemming:
        lst = []
        for tok in tokens:
            lst.append(stemmer.stem(tok))
        tokens = lst
    return tokens


def cosineSimilarity(query, index, DL):

    csDictionary = {}
    for term in query:
        postingList = read_posting_list(index, term)
        for id, tf in postingList:
            csDictionary[id] = csDictionary.get(id, 0) + tf

    for id, similatiry in csDictionary.items():
        csDictionary[id] = similatiry / (DL[id] * len(query))
    return csDictionary


def body_search(query, index, DL, titles):

    res = []
    lst_query = tokenize(query)
    terms = [t for t in index.posting_locs]
    filteredQ = [token for token in lst_query if token in terms]

    # calculating cosine Similarity
    csDictionary = cosineSimilarity(filteredQ, index, DL)

    # get top relevant 100
    list_dict = [(d_id, builtins.round(score, 5)) for d_id, score in csDictionary.items()]
    top_100 = sorted(list_dict, key=lambda x: x[1], reverse=True)[:100]

    for id, score in top_100:
        res.append((id, titles[id]))
    return res

def title_anchor_search(query, index, titles):

    queryTokens = tokenize(query)
    lst = []
    for term in queryTokens:
        try:
            lst.append(read_posting_list(index, term))
        except:
            lst.append([])

    counter = {}
    for posting in lst:
        for t in posting:
            if t[0] not in counter.keys():
                counter[t[0]] = 1
            else:
                counter[t[0]] = counter[t[0]] + 1
    items = list(counter.items())
    items.sort(key=lambda x: x[1], reverse=True)
    res = []
    for key, val in items:
        res.append((key, titles[key]))
    return res

################
# search
def get_relevant_docs(query, index, use_CosineSimilarity=False, DL=None):

    dictionary = {}
    try:
        if not use_CosineSimilarity:
            for token in query:
                for id, tf in read_posting_list(index, token):
                    dictionary[id] = dictionary.get(id, 0) + tf
        else:
            dictionary = cosineSimilarity(query, index, DL)
    except:
        pass
    return dictionary


def mainSearch(title_scores, body_scores, title_weight=0.7, text_weight=0.3, N=100):

    dictionary = {}
    body_lst = body_scores.keys()
    title_lst = title_scores.keys()
    all_docs_ids = set(list(title_scores.keys()) + list(body_scores.keys()))
    score = 0
    for doc in all_docs_ids:
        if doc in body_lst:
            if doc in title_lst:
                score = body_scores[doc] * text_weight + title_weight * title_scores[doc]
            else:
                score = body_scores[doc] * text_weight
        elif doc in title_lst:
            score = title_weight * title_scores[doc]
        dictionary[doc] = score

    res = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    res = res[:N]
    return res

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = tokenize(query)
    body_index = InvertedIndex.read_index("/content/bodyWithoutStem", "index")
    os.chdir("/content/bodyWithoutStem")
    relevant_text = get_relevant_docs(query, body_index)
    os.chdir("/content/titleWithoutStem")
    title_index = InvertedIndex.read_index("/content/titleWithoutStem", "index")
    relevant_title = get_relevant_docs(query, title_index)
    os.chdir("/content")
    results = mainSearch(relevant_title, relevant_text)
    for key, val in results:
        res.append((key, titleDict[key]))
    os.chdir("/content")
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    body_index = InvertedIndex.read_index("/content/bodyWithoutStem", "index")
    os.chdir("/content/bodyWithoutStem")
    res = body_search(query, body_index, DL, titleDict)
    os.chdir("/content")
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the titleWithStem. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a titleWithStem that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the titleWithStem (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    title_index = InvertedIndex.read_index("/content/titleWithoutStem", "index")
    os.chdir("/content/titleWithoutStem")
    res = title_anchor_search(query, title_index, titleDict)
    os.chdir("/content")

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    anchor_index = InvertedIndex.read_index("/content/anchorText", "index")
    os.chdir("/content/anchorText")
    res = title_anchor_search(query, anchor_index, titleDict)
    os.chdir("/content")
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    pageRank.columns = ['id', 'PageRank']
    ids = pageRank['id'].isin(wiki_ids)
    res = (pageRank.loc[ids])['PageRank'].tolist()
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for id in wiki_ids:
        try:
            res.append(pageView[id])
        except:
            res.append(0)
    # END SOLUTION
    return jsonify(res)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
