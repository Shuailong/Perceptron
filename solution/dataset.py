#!/usr/bin/env python
# encoding: utf-8

"""
dataset.py
 
Created by Shuailong on 2016-02-16.

Dataset for Assignment 1.

"""

import os,sys
import nltk
from nltk.corpus import stopwords
from collections import Counter
import pickle
from math import log

import featureextractor


BASE_PATH = '../data'
UNKNOWN_NOTATION = 'UNK'
get_feature = featureextractor.BoW

def get_vocab():
    '''
    :return: inverse document frequency(idf)
    :rtype: dict
    '''
    #try to get from pickle dump
    if os.path.isfile('idf.p'):
        idf = pickle.load(open('idf.p', 'rb'))
        return idf

    idf = {}

    D = 0 # total number of documents

    data_sets = ['train/atheism', 'train/politics', 'train/science', 'train/sports']
    
    stops = set(stopwords.words('english'))

    vocab = []
    for data_set in data_sets:
        root = os.path.join(BASE_PATH, data_set)
        for file in os.listdir(root):
            file_dir = os.path.join(root, file)
            D += 1
            with open(file_dir) as f:
                content = f.read()
            # tokenization
            words = []
            try:
                words = nltk.word_tokenize(content)
            except Exception, e:
                # ignore decoding errors
                pass
            # remove stop words
            words = [word.lower() for word in words if word not in stops]

            for word in words:
                vocab.append(word)
    
    #remove low freq. word and add 'unkown' word
    freq = Counter(vocab)
    freq[UNKNOWN_NOTATION] = 0
    for word in freq:
        if freq[word] > 1:
            idf[word] = log(D/float(freq[word]))
        else:
            freq[UNKNOWN_NOTATION] += freq[word]
    idf[UNKNOWN_NOTATION] = log(D/float(freq[UNKNOWN_NOTATION]))
    
    pickle.dump(idf, open('idf.p', 'wb'))

    return idf


def get_atheism_train_data():
    """
    :return: features of training data
    :rtype: List of List
    """
    #try to get from pickle dump
    if os.path.isfile('atheism_train.p'):
        data = pickle.load(open('atheism_train.p', 'rb'))
        return data

    root = os.path.join(BASE_PATH, 'train/atheism')
    data = []
    for file in os.listdir(root):
        file_dir = os.path.join(root, file)
        with open(file_dir) as f:
            content = f.read()
        feature = get_feature(content)
        data.append(feature)

    pickle.dump(data, open('atheism_train.p', 'wb'))
    return data

def get_politics_train_data():
    """
    :return: features of training data
    :rtype: List of List
    """
    #try to get from pickle dump
    if os.path.isfile('politics_train.p'):
        data = pickle.load(open('politics_train.p', 'rb'))
        return data

    root = os.path.join(BASE_PATH, 'train/politics')
    data = []
    for file in os.listdir(root):
        file_dir = os.path.join(root, file)
        with open(file_dir) as f:
            content = f.read()
        feature = get_feature(content)
        data.append(feature)

    pickle.dump(data, open('politics_train.p', 'wb'))
    return data

def get_science_train_data():
    """
    :return: features of training data
    :rtype: List of List
    """
    #try to get from pickle dump
    if os.path.isfile('science_train.p'):
        data = pickle.load(open('science_train.p', 'rb'))
        return data
    root = os.path.join(BASE_PATH, 'train/science')
    data = []
    for file in os.listdir(root):
        file_dir = os.path.join(root, file)
        with open(file_dir) as f:
            content = f.read()
        feature = get_feature(content)
        data.append(feature)

    pickle.dump(data, open('science_train.p', 'wb'))

    return data

def get_sports_train_data():
    """
    :return: features of training data
    :rtype: List of List
    """
    #try to get from pickle dump
    if os.path.isfile('sports_train.p'):
        data = pickle.load(open('sports_train.p', 'rb'))
        return data

    root = os.path.join(BASE_PATH, 'train/sports')
    data = []
    for file in os.listdir(root):
        file_dir = os.path.join(root, file)
        with open(file_dir) as f:
            content = f.read()
        feature = get_feature(content)
        data.append(feature)

    pickle.dump(data, open('sports_train.p', 'wb'))

    return data


def get_atheism_test_data():
    """
    :return: features of training data
    :rtype: List of List
    """
    #try to get from pickle dump
    if os.path.isfile('atheism_test.p'):
        data = pickle.load(open('atheism_test.p', 'rb'))
        return data

    root = os.path.join(BASE_PATH, 'test/atheism')
    data = []
    for file in os.listdir(root):
        file_dir = os.path.join(root, file)
        with open(file_dir) as f:
            content = f.read()
        feature = get_feature(content)
        data.append(feature)

    pickle.dump(data, open('atheism_test.p', 'wb'))
    return data

def get_politics_test_data():
    """
    :return: features of training data
    :rtype: List of List
    """
    #try to get from pickle dump
    if os.path.isfile('politics_test.p'):
        data = pickle.load(open('politics_test.p', 'rb'))
        return data

    root = os.path.join(BASE_PATH, 'test/politics')
    data = []
    for file in os.listdir(root):
        file_dir = os.path.join(root, file)
        with open(file_dir) as f:
            content = f.read()
        feature = get_feature(content)
        data.append(feature)

    pickle.dump(data, open('politics_test.p', 'wb'))
    return data

def get_science_test_data():
    """
    :return: features of training data
    :rtype: List of List
    """
    #try to get from pickle dump
    if os.path.isfile('science_test.p'):
        data = pickle.load(open('science_test.p', 'rb'))
        return data

    root = os.path.join(BASE_PATH, 'test/science')
    data = []
    for file in os.listdir(root):
        file_dir = os.path.join(root, file)
        with open(file_dir) as f:
            content = f.read()
        feature = get_feature(content)
        data.append(feature)

    pickle.dump(data, open('science_test.p', 'wb'))

    return data

def get_sports_test_data():
    """
    :return: features of training data
    :rtype: List of List
    """
    #try to get from pickle dump
    if os.path.isfile('sports_test.p'):
        data = pickle.load(open('sports_test.p', 'rb'))
        return data

    root = os.path.join(BASE_PATH, 'test/sports')
    data = []
    for file in os.listdir(root):
        file_dir = os.path.join(root, file)
        with open(file_dir) as f:
            content = f.read()
        feature = get_feature(content)
        data.append(feature)
        
    pickle.dump(data, open('sports_test.p', 'wb'))

    return data

def main():
    # print get_vocab()

    # print get_atheism_train_data()[0]
    # print get_atheism_test_data()[0]
    # print get_politics_train_data()[0]
    # print get_politics_test_data()[0]
    # print get_science_train_data()[0]
    # print get_science_test_data()[0]
    # print get_sports_train_data()[0]
    # print get_sports_test_data()[0]
    pass

if __name__ == '__main__':
    main()
    
