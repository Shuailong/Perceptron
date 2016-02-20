#!/usr/bin/env python
# encoding: utf-8

"""
perceptron.py
 
Created by Shuailong on 2016-02-16.

Basic/Averaged perceptron algorithm for Assignment 1, part (2).

"""

import os
import pickle


def train(X, y, vocab_size):
    '''
    X: Feature vectors
    y: labels, List of Booleans
    rtype: List
    return: theta and theta0
    '''
    n = len(X)
    d = vocab_size
    theta = [0]*d
    theta0 = 0

    # averaged version
    # http://www.ciml.info/dl/v0_8/ciml-v0_8-ch03.pdf
    u = [0]*d
    beta = 0
    c = 1

    # while True:
    for l in range(2000):
        count = 0
        for j in range(n):
            x = X[j]
            sumation = 0
            for k in range(d):
                sumation += x.get(k,0)*theta[k]
            sumation += theta0

            if sumation*y[j] <= 0:
                count += 1
                for k in range(d):
                    theta[k] += y[j]*x.get(k,0)
                    u[k] += y[j]*x.get(k,0)*c
                theta0 += y[j]
                beta += y[j]*c
            c += 1

        print str(l) + 'th iteration, mislabeled: ' + str(count) + '.'
        if count == 0:
            break

    # for k in range(len(theta)):
    #     theta[k] -= u[k]/float(c)

    # theta0 -= beta/float(c)

    return (theta, theta0)
    
if __name__ == '__main__':
    pass
    
