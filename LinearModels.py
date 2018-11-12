#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:14:46 2017

@author: aditya
"""


from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
import numpy as np



class TLearner(object):
    def __init__(self, alpha = 0.01):
        super().__init__()
        self.modelT = Ridge(alpha = alpha)
        self.modelC = Ridge(alpha = alpha)
        self.fitted = False
             
        
    def fit(self, X, Y, A):
        Xt, Xc, Yt, Yc = X[A], X[~A], Y[A], Y[~A]
        self.modelT.fit(Xt, Yt)
        self.modelC.fit(Xc, Yc)
        self.fitted = True
        
    def predict(self, X, switch = False):
        assert self.fitted
        delta = self.modelT.predict(X) - self.modelC.predict(X)
        
        # want to target people with high negative values
        delta = delta if switch else -delta
        return delta

class XLearner(TLearner):
    def __init__(self, alpha = 0.01):
        super().__init__()
        self.modelTX = Ridge(alpha = alpha)
        self.modelCX = Ridge(alpha = alpha)
        self.propensity = LogisticRegression()
        
    def __propensityFit(self, Xt, Xc):
        At, Ac = np.ones(len(Xt)), np.zeros(len(Xc))
        self.propensity.fit(np.concatenate((Xt, Xc)), np.concatenate((At, Ac)))
    
    def fit(self, X, Y, A):
        Xt, Xc, Yt, Yc = X[A], X[~A], Y[A], Y[~A]
        self.__propensityFit(Xt, Xc)
        self.modelT.fit(Xt, Yt)
        self.modelC.fit(Xc, Yc)
        Rt = Yt -self.modelC.predict(Xt)
        Rc = self.modelT.predict(Xc) - Yc
        self.modelTX.fit(Xt, Rt)
        self.modelCX.fit(Xc, Rc)
        self.fitted = True
    
    def predict(self, X, switch = False):
        assert self.fitted
        weight = self.propensity.predict_proba(X)
        delta = weight[:,1]*self.modelTX.predict(X) + weight[:,0]*self.modelCX.predict(X)
        
        # want to target people with high negative values
        delta = delta if switch else -delta
        return delta
    
    
class ZLearner(object):
    def __init__(self, alpha = 0.01):
        super().__init__()
        self.model = Ridge(alpha = alpha)
        
    def fit(self, X, Y, A):
        Xt, Xc, Yt, Yc = X[A], X[~A], Y[A], Y[~A]
        X, Y = np.concatenate((Xt, Xc)), np.concatenate((Yt, -Yc))
        self.model.fit(X, Y)
        self.fitted = True
    
    def predict(self, X, switch = False):
        assert self.fitted
        delta = self.model.predict(X)
        # want to target people with high negative values
        delta = delta if switch else -delta
        return delta


class ProgLearner(object):
    def __init__(self, alpha = 0.01):
        super().__init__()
        self.model = Ridge(alpha = alpha)
        
    def fit(self, X, Y, A):
        self.model.fit(X, Y)
        self.fitted = True
    
    def predict(self, X, switch = False):
        assert self.fitted
        delta = self.model.predict(X)
        # want to target people with high negative values
        return delta
