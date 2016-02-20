#!/usr/bin/env python
# encoding: utf-8

"""
gradientdescent.py
 
Created by Shuailong on 2016-02-16.

Stochastic Gradient Descent algorithm for Assignment 1, part (3,5).

"""

import os
import pickle
from random import randint

def loss(X, y, theta, theta0, vocab_size):
    '''
    Hinge loss.
    X: Feature vectors
    y: labels, List of Booleans
    theta, theta0: model params
    return: loss
    rtype: float
    '''
    loss = 0
    n = len(X)
    d = vocab_size
    for j in range(n):
        x = X[j]
        sumation = 0
        for k in range(d):
            sumation += x.get(k,0)*theta[k]*y[j]
        sumation += theta0*y[j]

        loss += max(sumation, 0)
    return loss/float(n)

# part (5)
def train_regularize(X, y, lamda, vocab_size):
    '''
    Gradient Descent training with regularization term
    X: Feature vectors
    y: labels, List of Booleans
    rtype: List
    return: theta and theta0
    '''
    n = len(X)
    d = vocab_size
    theta = [0]*d
    theta0 = 0

    best_theta = theta
    best_theta0 = theta0
    best_loss = float('inf')

    last_loss = 0

    # while True:
    for l in range(200000):
        # ita = 1/float(l+1)
        ita = 0.01
        j = randint(0, n-1)
        x = X[j]
        sumation = 0
        for k in range(d):
            sumation += x.get(k,0)*theta[k]
        sumation += theta0

        if sumation*y[j] <= 1:
            for k in range(d):
                theta[k] += y[j]*x.get(k,0)*ita
            theta0 += y[j]*ita

        if l % 1000 == 0:
            print l, 'loss: ', loss(X, y, theta, theta0, vocab_size)

            lo = loss(X, y, theta, theta0, vocab_size)
            if lo < best_loss:
                best_theta = theta
                best_theta0 = theta0
                best_loss = lo
        
            if last_loss == lo:
                break
                
        last_loss = lo

    return (theta, theta0)

# part (3)
def train(X, y, vocab_size):
    '''
    Gradient Descent training
    X: Feature vectors
    y: labels, List of Booleans
    rtype: List
    return: theta and theta0
    '''
    n = len(X)
    d = vocab_size
    theta = [0]*d
    theta0 = 0

    best_theta = theta
    best_theta0 = theta0
    best_loss = float('inf')

    last_loss = 0

    # while True:
    for l in range(200000):
        # ita = 1/float(l+1)
        ita = 0.01
        j = randint(0, n-1)
        x = X[j]
        sumation = 0
        for k in range(d):
            sumation += x.get(k,0)*theta[k]
        sumation += theta0

        if sumation*y[j] <= 1:
            for k in range(d):
                theta[k] += y[j]*x.get(k,0)*ita
            theta0 += y[j]*ita

        if l % 1000 == 0:
            print l, 'loss: ', loss(X, y, theta, theta0, vocab_size)

            lo = loss(X, y, theta, theta0, vocab_size)
            if lo < best_loss:
                best_theta = theta
                best_theta0 = theta0
                best_loss = lo
        
            if last_loss == lo:
                break
                
        last_loss = lo

    return (theta, theta0)
    
if __name__ == '__main__':
    pass
    
