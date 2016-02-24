#!/usr/bin/env python
# encoding: utf-8

"""
featureextractor.py
 
Created by Shuailong on 2016-02-24.

Feature extractor for Assignment 1.

"""

import nltk
from nltk.corpus import stopwords
from collections import Counter
from math import log

import dataset

def tfidf(content):
    """
    :content: an article
    :type str: string
    :return: the feature vector extracted from an article.
    :rtype: dict{word: tfidf}
    """
    idf = dataset.get_vocab()
    vocab = idf.keys()
    words = []
    try:
        words = nltk.word_tokenize(content)
    except Exception, e:
        pass

    stops = set(stopwords.words('english'))
    # remove stop words
    words = [word.lower() for word in words if word not in stops]
    freq = Counter(words)
    tf = {} # term frequency
    known_words = []
    for i in range(len(vocab)):
        if freq.get(vocab[i], 0) != 0:
            known_words.append(vocab[i])
            tf[vocab[i]] = 1 + log(freq[vocab[i]])
            
    unkown_words = []
    for word in freq:
        if word not in known_words:
            unkown_words.append(word)
    if len(unkown_words) > 0:
        tf[dataset.UNKNOWN_NOTATION] = 1 + log(len(unkown_words))

    feature = {}
    for i in range(len(vocab)):
        tfidf = tf.get(vocab[i],0) * idf.get(vocab[i], 0)
        if tfidf != 0:
            feature[i] = tfidf

    return feature

def BoW(content):
    """
    :content: an article
    :type str: string
    :return: the feature vector extracted from an article.
    :rtype: dict{word: tf}
    """
    idf = dataset.get_vocab()
    vocab = idf.keys()
    words = []
    try:
        words = nltk.word_tokenize(content)
    except Exception, e:
        pass

    stops = set(stopwords.words('english'))
    # remove stop words
    words = [word.lower() for word in words if word not in stops]
    freq = Counter(words)
    tf = {}
    known_words = []
    for i in range(len(vocab)):
        if freq.get(vocab[i], 0) != 0:
            known_words.append(vocab[i])
            tf[i] = freq[vocab[i]]
    unkown_words = []
    for word in freq:
        if word not in known_words:
            unkown_words.append(word)
    unknown_index = vocab.index(dataset.UNKNOWN_NOTATION)
    tf[unknown_index] = len(unkown_words)

    return tf


