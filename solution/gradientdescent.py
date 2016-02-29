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
from random import shuffle

def train(X, y, lamda, ita, vocab_size):
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

    orders = range(n)
    # while True:
    for l in range(100):
        # ita = 1/float(l+1)
        shuffle(orders)
        loss = 0
        for j in orders: 
            x = X[j]

            sumation = 0
            for k in range(d):
                sumation += x.get(k,0)*theta[k]
            sumation += theta0

            if sumation*y[j] <= 1:
                for k in range(d):
                    theta[k] += (-2*lamda*ita)*theta[k] + y[j]*x.get(k,0)*ita
                theta0 += y[j]*ita

            loss += max(sumation, 0)
        loss /= float(n)
        print l, 'loss: ', loss

        if loss < best_loss:
            best_theta = theta
            best_theta0 = theta0
            best_loss = loss
    
        if last_loss - loss >= 0 and last_loss - loss < 0.001:
            break 

        last_loss = loss

    return (theta, theta0)

if __name__ == '__main__':
    pass
    
