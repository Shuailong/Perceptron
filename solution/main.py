#!/usr/bin/env python
# encoding: utf-8

"""
main.py
 
Created by Shuailong on 2016-02-18.

Main program entry.

"""

import dataset
import time
import os
import pickle

import perceptron
import gradientdescent
from random import shuffle


def predict(theta, theta0, X_test, vocab_size):
    '''
    theta, theta0: model params
    X_test: test data
    rtype: List of Boolean
    return: classification result
    '''
    n = len(X_test)
    d = vocab_size
    res = [0]*n

    for j in range(n):
        x = X_test[j]
        sumation = 0
        for k in range(d):
            sumation += x.get(k,0)*theta[k]
        sumation += theta0
        if sumation > 0:
            res[j] = 1
        else:
            res[j] = -1

    return res

def multipredict(thetas, X, vocab_size):
    '''
    thetas: params for all four classifers
    X: data to be classified
    vocab_size: vocabulary size
    '''
    # encoding: atheism = 1, sports = 2, politics = 3, science = 4

    theta_atheism, theta0_atheism = thetas[0]
    theta_sports, theta0_sports = thetas[1]
    theta_politics, theta0_politics = thetas[2]
    theta_science, theta0_science = thetas[3]

    n = len(X)
    d = vocab_size
    res = [0]*n
    for j in range(n):
        x = X[j]
        best_summation = 0
        for i in range(len(thetas)):
            theta = thetas[i][0]
            theta0 = thetas[i][1]
            sumation = 0
            for k in range(d):
                sumation += x.get(k,0)*theta[k]
            sumation += theta0

            if res[j] == 0:
                res[j] = i+1
            elif sumation > best_summation:
                res[j] = i+1
                best_summation = sumation

    return res


def score(predict, true):
    '''
    predict: prediction result, List of Booleans
    true: true labels, List of Booleans
    rtype: float
    return: accuracy
    '''
    if len(predict) != len(true):
        raise Exception('Length mismatch!')
    correct = 0
    for i in range(len(predict)):
        if predict[i] == true[i]:
            correct += 1
    
    return correct/float(len(predict))

def multiscore(predict, true):
    # use precision and recall?
    return score(predict, true)

# part(2,3)
def binary_classify():
    print 'Building vocabulary...'
    vocab = dataset.get_vocab().keys()
    vocab_size = len(vocab)
    
    print 'Loading data...'

    atheism_train = dataset.get_atheism_train_data()
    atheism_test = dataset.get_atheism_test_data()
    sports_train = dataset.get_sports_train_data()
    sports_test = dataset.get_sports_test_data()

    print 'Data loaded. ' + str(len(atheism_train)) + ' atheism_train tuples, ' \
        + str(len(atheism_test)) + ' atheism_test tuples, ' + str(len(sports_train)) + ' sports_train tuples, '\
        + str(len(sports_test)) + ' sports_test tuples.'

    # assume: atheism = -1, sports = 1
    X = atheism_train + sports_train
    y = [-1]*len(atheism_train) + [1]*len(sports_train)

    print 'Start training...'

    # choose train model
    # train = perceptron.train
    # if os.path.isfile('perceptron_model.p'):
    #     theta, theta0 = pickle.load(open('perceptron_model.p', 'rb'))
    # else:
    #     theta, theta0 = train(X, y, vocab_size)
    #     pickle.dump((theta, theta0), open('perceptron_model.p', 'wb'))

    itas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    for ita in itas:
        train = gradientdescent.train
        theta, theta0 = train(X, y, ita, vocab_size)
        print 'Training ended.'

        X_test = atheism_test + sports_test
        y_test = [-1]*len(atheism_test) + [1]*len(sports_test)

        print 'Start predicting...'
        predict_train = predict(theta, theta0, X, vocab_size)
        score_train = score(predict_train, y)
        predict_test = predict(theta, theta0, X_test, vocab_size)
        score_test = score(predict_test, y_test)
        print 'Predicting ended.'
        # print 'ita: ', ita
        print 'Train: ', str(round(score_train*100, 2)) + '%'
        print 'Test: ', str(round(score_test*100, 2)) + '%'

# part(5)
def binary_classify_regularize():
    vocab = dataset.get_vocab().keys()
    vocab_size = len(vocab)
    
    print 'Loading data...'

    atheism_train = dataset.get_atheism_train_data()
    atheism_test = dataset.get_atheism_test_data()
    sports_train = dataset.get_sports_train_data()
    sports_test = dataset.get_sports_test_data()

    print 'Data loaded. ' + str(len(atheism_train)) + ' atheism_train tuples, ' \
        + str(len(atheism_test)) + ' atheism_test tuples, ' + str(len(sports_train)) + ' sports_train tuples, '\
        + str(len(sports_test)) + ' sports_test tuples.'

    # assume: atheism = -1, sports = 1
    X = atheism_train + sports_train
    y = [-1]*len(atheism_train) + [1]*len(sports_train)

    print 'Start training...'

    train = gradientdescent.train
    lambdas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    ita = 0.001
    for lamda in lambdas:
        print 'Lambda: ', lamda
        theta, theta0 = train(X, y, lamda, ita, vocab_size)
        pickle.dump((theta, theta0), open('gradient_regularize_model.p', 'wb'))
        print 'Training ended.'

        X_test = atheism_test + sports_test
        y_test = [-1]*len(atheism_test) + [1]*len(sports_test)

        print 'Start predicting...'
        predict_train = predict(theta, theta0, X, vocab_size)
        score_train = score(predict_train, y)
        predict_test = predict(theta, theta0, X_test, vocab_size)
        score_test = score(predict_test, y_test)
        print 'Predicting ended.'
        print 'Lambda:', lamda
        print 'Train: ', str(round(score_train*100, 2)) + '%.'
        print 'Test: ', str(round(score_test*100, 2)) + '%.'

#part (4)
def multiclassify():
    vocab = dataset.get_vocab().keys()
    vocab_size = len(vocab)
    
    print 'Loading data...'

    atheism_train = dataset.get_atheism_train_data()
    atheism_test = dataset.get_atheism_test_data()
    sports_train = dataset.get_sports_train_data()
    sports_test = dataset.get_sports_test_data()
    politics_train = dataset.get_politics_train_data()
    politics_test = dataset.get_politics_test_data()
    science_train = dataset.get_science_train_data()
    science_test = dataset.get_science_test_data()

    print 'Data loaded. '
    print str(len(atheism_train)) + ' atheism_train tuples;'
    print str(len(atheism_test)) + ' atheism_test tuples;'
    print str(len(sports_train)) + ' sports_train tuples;'
    print str(len(sports_test)) + ' sports_test tuples;'    
    print str(len(politics_train)) + ' politics_train tuples;'
    print str(len(politics_test)) + ' politics_test tuples;'
    print str(len(science_train)) + ' science_train tuples;'
    print str(len(science_test)) + ' science_test tuples.'

    train = gradientdescent.train
    ita = 0.001
    lamda = 0.001

    print 'Start training atheism vs others...'

    # assume atheism = 1, others = -1
    X = atheism_train + sports_train + politics_train + science_train
    y = [1]*len(atheism_train) + [-1]*(len(sports_train) + len(politics_train) + len(science_train)) 
    # random pick len(atheism_train) data from -1 labelled data to void skewed data
    # shuffle(sports_train)
    # shuffle(politics_train)
    # shuffle(science_train)
    # X = atheism_train + sports_train[:len(sports_train)/3] + politics_train[:len(politics_train)/3] + science_train[:len(science_train)/3]
    # y = [1]*len(atheism_train) + [-1]*(len(sports_train)/3+len(politics_train)/3+len(science_train)/3)
    if os.path.isfile('atheism_vs_others.p'):
        theta_atheism, theta0_atheism = pickle.load(open('atheism_vs_others.p', 'rb'))
    else:
        theta_atheism, theta0_atheism = train(X, y, lamda, ita, vocab_size)
        pickle.dump((theta_atheism, theta0_atheism), open('atheism_vs_others.p', 'wb'))
    print 'Training atheism vs others ended.'

    print 'Start training politics vs others...'
    X = politics_train + sports_train + atheism_train + science_train
    y = [1]*len(politics_train) + [-1]*(len(sports_train) + len(atheism_train) + len(science_train))
    # shuffle(sports_train)
    # shuffle(atheism_train)
    # shuffle(science_train)
    # X = politics_train + sports_train[:len(sports_train)/3] + atheism_train[:len(atheism_train)/3] + science_train[:len(science_train)/3]
    # y = [1]*len(politics_train) + [-1]*(len(sports_train)/3+len(atheism_train)/3+len(science_train)/3)
    if os.path.isfile('politics_vs_others.p'):
        theta_politics, theta0_politics = pickle.load(open('politics_vs_others.p', 'rb'))
    else:
        theta_politics, theta0_politics = train(X, y, lamda, ita, vocab_size)
        pickle.dump((theta_politics, theta0_politics), open('politics_vs_others.p', 'wb'))
    print 'Training politics vs others ended.'

    print 'Start training sports vs others...'
    X = science_train + sports_train + atheism_train + politics_train
    y = [1]*len(science_train) + [-1]*(len(sports_train) + len(atheism_train) + len(politics_train))
    # shuffle(sports_train)
    # shuffle(atheism_train)
    # shuffle(politics_train)
    # X = science_train + sports_train[:len(sports_train)/3] + atheism_train[:len(atheism_train)/3] + politics_train[:len(politics_train)/3]
    # y = [1]*len(science_train) + [-1]*(len(sports_train)/3+len(atheism_train)/3+len(politics_train)/3)
    if os.path.isfile('science_vs_others.p'):
        theta_science, theta0_science = pickle.load(open('science_vs_others.p', 'rb'))
    else:
        theta_science, theta0_science = train(X, y, lamda, ita, vocab_size)
        pickle.dump((theta_science, theta0_science), open('science_vs_others.p', 'wb'))
    print 'Training science vs others ended.'

    print 'Start training science vs others...'
    X = sports_train + science_train + atheism_train + politics_train
    y = [1]*len(sports_train) + [-1]*(len(science_train) + len(atheism_train) + len(politics_train))  
    # shuffle(science_train)
    # shuffle(atheism_train)
    # shuffle(politics_train)
    # X = sports_train + science_train[:len(science_train)/3] + atheism_train[:len(atheism_train)/3] + politics_train[:len(politics_train)/3]
    # y = [1]*len(sports_train) + [-1]*(len(science_train)/3+len(atheism_train)/3+len(politics_train)/3)  
    if os.path.isfile('sports_vs_others.p'):
        theta_sports, theta0_sports = pickle.load(open('sports_vs_others.p', 'rb'))
    else:
        theta_sports, theta0_sports = train(X, y, lamda, ita, vocab_size)
        pickle.dump((theta_sports, theta0_sports), open('sports_vs_others.p', 'wb'))
    print 'Training sports vs others ended.'

    print 'Start predicting...'
    
    # encoding: atheism = 1, sports = 2, politics = 3, science = 4
    X_train = atheism_train + sports_train + politics_train + science_train
    y_train = [1]*len(atheism_train) + [2]*len(sports_train) + [3]*len(politics_train) + [4]*len(science_train)
    
    X_test = atheism_test + sports_test + politics_test + science_test
    y_test = [1]*len(atheism_test) + [2]*len(sports_test) + [3]*len(politics_test) + [4]*len(science_test)

    thetas = [(theta_atheism, theta0_atheism),(theta_sports, theta0_sports),(theta_politics, theta0_politics),(theta_science, theta0_science)]
    predict_train = multipredict(thetas, X_train, vocab_size)
    score_train = multiscore(predict_train, y_train)
    predict_test = multipredict(thetas, X_test, vocab_size)
    score_test = multiscore(predict_test, y_test)
    print 'Predicting ended.'

    print 'Train: ', str(round(score_train*100, 2)) + '%'
    print 'Test: ', str(round(score_test*100, 2)) + '%'


def main():
    start_time = time.time()

    # binary_classify()
    # multiclassify()
    binary_classify_regularize()

    print '----------' + str(round(time.time() - start_time, 2)) + ' seconds.----------'


if __name__ == '__main__':
    main()
    
