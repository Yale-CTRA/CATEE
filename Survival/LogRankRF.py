# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 09:54:45 2018

@author: adityabiswas
"""


import numpy as np
from joblib import Parallel, delayed
import time
from copy import copy

from scipy import stats

class SurvStats(object):
    # 3 scenarios: neither (0), branch w/ future removals (1) branch w/ future insertions (2)
    # removeData should be set for scenario 1 and outcomes should be all; allTimes should be scenario 2
    def __init__(self, outcomes, intervention, allTimes = None):
        super().__init__()
        self.avgEventTime = 605

        if allTimes is None:
            allTimes = outcomes[:,1]
        # statistics to track population
        self.times = np.unique(allTimes)
        m = len(self.times)
        self.nt, self.nc = np.zeros(m, dtype = np.int64), np.zeros(m, dtype = np.int64)
        self.dt, self.dc = np.zeros(m, dtype = np.int64), np.zeros(m, dtype = np.int64)
        self.totalT, self.totalC = 0, 0
                
        # initialize with current data
        for i in range(len(outcomes)):
            y, t, a = outcomes[i, 0], outcomes[i, 1], intervention[i]
            self.insert(y, t, a)
    
    def update(self, y, t, a, shift):
        index = self.where(t)
        if a:
            self.totalT += shift
            self.nt[:index+1] += shift
            if y == 1:
                self.dt[index] += shift
        else:
            self.nc[:index+1] += shift
            self.totalC += shift
            if y == 1:
                self.dc[index] += shift
                
    def insert(self, y, t, a):
        self.update(y, t, a, shift = 1)
            
    def remove(self, y, t, a):
        self.update(y, t, a, shift = -1)        
        
    def where(self, t):
        index = np.where(self.times == t)[0][0]
        return index       
    
    def getStatistic(self):
        n = self.nc + self.nt
        d = self.dc + self.dt
        select = np.logical_and(np.logical_and(d > 0, n-d > 0), 
                                np.logical_and(self.nt > 0, self.nc > 0))
        val = 0
        if np.sum(select) > 0:
            n = n[select]
            d = d[select]
            nt, nc = self.nt[select], self.nc[select]
            dc = self.dc[select]
            
            num = np.sum(dc - d*nc/n)
            den = np.power(np.sum(nc*nt*d*(n-d)/(n*n*(n-1))), 0.5)
            val = num/den
            if np.isnan(val):
                print(n)
                print(d)
                print(nc)
                print(nt)
                val = 0
        
        return val
    
        
    def getLogRank(self, twoSided = True):
        val = self.getStatistic()
        if twoSided:
            pval = stats.norm.sf(abs(val))*2
        else:
            pval = stats.norm.sf(val)
        return pval


    def getRMSTdiff(self, tau = 365.25*4):
        select_T = np.logical_and(self.dt > 0, self.times < tau)
        select_C = np.logical_and(self.dc > 0, self.times < tau)

        times_T, nt, dt = self.times[select_T], self.nt[select_T], self.dt[select_T]
        times_C, nc, dc = self.times[select_C], self.nc[select_C], self.dc[select_C]   

        # check if anybody died before tau in this population
        if np.any(select_T):
            S = np.cumprod(1 - dt/nt)
            deltas = times_T[1:] - times_T[:-1]
            # add first and last rectangles separately
            area_T = np.sum(S[:-1]*deltas) + times_T[0] + S[-1]*(tau - times_T[-1])

        else:
            area_T = tau
            
        # check if anybody died before tau in this population
        if np.any(select_C):
            S = np.cumprod(1 - dc/nc)
            deltas = times_C[1:] - times_C[:-1]
            # add first and last rectangles separately
            area_C = np.sum(S[:-1]*deltas) + times_C[0] + S[-1]*(tau - times_C[-1])

        else:
            area_C = tau
            
        return area_T - area_C
        


class RandomForest(object):
    def __init__(self, numTrees, numTry, bootstrapPercent, minGroup, pval, verbose = True):
        super().__init__()
        self.numTrees = numTrees
        self.numTry = numTry
        self.bootstrapPercent = bootstrapPercent
        self.minGroup = minGroup
        self.pval = pval
        self.verbose = verbose
        self.fitted = False
        self.counter = 0
        
    def makeTree(self, data, intervention, outcomes, bootstrapCut):
        np.random.seed()
        sorter = np.random.permutation(np.arange(len(data)))
        data, intervention, outcomes = data[sorter,...], intervention[sorter], outcomes[sorter,...]
        tree = Root(data[:bootstrapCut,...], intervention[:bootstrapCut], 
                    outcomes[:bootstrapCut,...], self.numTry, self.minGroup, self.pval, self.columnTypes)
        self.counter += 1
        return tree

    def fit(self, data, intervention, outcomes):
        ## data is predictor float matrix
        ## intervention is boolean array
        ## outcomes is matrix with Y in col 0 and T in col 1
        numRows, self.numColumns = np.shape(data)
        bootstrapCut = int(np.round(numRows*self.bootstrapPercent))
        self.recordColumnTypes(data)
        
        ## run training over all trees
        if self.verbose:
            verbose = 12
        else:
            verbose = 0
        self.trees = Parallel(n_jobs = -1, backend = 'multiprocessing', verbose = verbose)(delayed(self.makeTree)(data, intervention, 
                              outcomes, bootstrapCut) for i in range(self.numTrees))
        self.fitted = True
        
    def predict(self, data):
        assert self.fitted
        numRows = len(data)
        allResults = np.zeros((numRows, self.numTrees))
#        weights = np.zeros((numRows, self.numTrees))
#        
#        for i in range(self.numTrees):
#            results, totals = self.trees[i].predict(data)
#            allResults[:,i] = results
#            weights[:,i] = totals
        
#        weightSums = np.sum(weights[:,1])
#        weights = np.apply_along_axis(lambda x: x/weightSums, axis = 1, arr =  weights)
#        allResults = np.sum(weights*allResults, axis = 1)
        return np.mean(allResults, axis = 1)
    
    def getNumLeaves(self):
        numLeaves = 0
        for i in range(self.numTrees):
            numLeaves += self.trees[i].getNumLeaves()
        return numLeaves/self.numTrees
    
    def getVarImportances(self):
        assert self.fitted
        varImportances = np.zeros(self.numColumns)
        for i in range(self.numTrees):
            varImportances = varImportances + self.trees[i].varImportances
        varImportances = varImportances/self.numTrees
        return varImportances
    
    def recordColumnTypes(self, data):
        self.columnTypes = [np.dtype(np.float) for i in range(self.numColumns)]
        for i in range(self.numColumns):
            ## cutoff of 3 used because mean imputation is used creating a 3rd unique value for boolean vars
            if len(np.unique(data[:100,i])) <= 3 :
                self.columnTypes[i] = np.dtype(np.bool)

            

class Root(object):
    def __init__(self,  data, intervention, outcomes, numTry, minGroup, pval, columnTypes):
        super().__init__()
        self.numRows, self.numColumns = np.shape(data)
        self.ID = np.arange(self.numRows)
        self.data = data
        self.intervention = intervention
        self.outcomes = outcomes
        self.columnTypes = columnTypes
        self.numTry = numTry
        self.minGroup = minGroup
        self.pval = pval
        self.n = len(data)
        self.varImportances = np.zeros(self.numColumns)
        
        self.tree = Node(self, self.ID)
        
        ## free memory
        self.data, self.ID, self.intervention, self.outcomes = None, None, None, None
    
    def predict(self, data):
        self.data = data
        self.numRows = len(data)
        self.ID = np.arange(self.numRows)
        self.results = np.zeros(self.numRows)
        self.totals = np.zeros(self.numRows)
        self.tree.predict(self.ID)
        ## free memory
        self.data, self.numRows, self.ID = None, None, None
        return self.results, self.totals
    
    def getNumLeaves(self):
        self.numLeaves = 0
        self.tree.countLeaf()
        return self.numLeaves
        
        

class Node(object):
    def __init__(self, root, rowIndex, maximize = True):
        super().__init__()
        self.root = root
        self.rowIndex = rowIndex
        ## peformance variable at this value indicates no appropriate split was found
        self.maximize = maximize
        self.noPerformanceMarker = -99999999999
        
        ## if leaf
        self.isLeaf = False
        self.leafValue = None
        
        ## if non-leaf
        self.leftChild, self.rightChild = None, None
        self.column, self.splitPoint = None, None
        
        
        self.fit()
    
    def predict(self, rowIndex):
        if self.isLeaf:
            self.root.results[rowIndex] = self.leafValue
            self.root.totals[rowIndex] = self.leafTotal
        else:
            L_ID, R_ID = self.splitIDs(rowIndex)
            self.leftChild.predict(L_ID)
            self.rightChild.predict(R_ID)
    
    def countLeaf(self):
        if self.isLeaf:
            self.root.numLeaves += 1
        else:
            self.leftChild.countLeaf()
            self.rightChild.countLeaf()
    
    def splitIDs(self, rowIndex):
        data, ID = self.root.data[rowIndex,self.column], self.root.ID[rowIndex]
        L_select = data <= self.splitPoint
        R_select = np.logical_not(L_select)
        L_ID, R_ID = ID[L_select], ID[R_select]
        return L_ID, R_ID
    
    def makeLeaf(self):
        self.isLeaf = True
        outcomes = self.root.outcomes[self.rowIndex,...]
        intervention = self.root.intervention[self.rowIndex]
        model = SurvStats(outcomes, intervention)

        self.leafValue = model.getStatistic()
        self.leafTotal = model.totalC + model.totalT
        
    
    def fit(self):
        column, splitPoint, performance = self.evaluateSplits()
        if column is None:
            self.makeLeaf()
        else:
            self.column, self.splitPoint = column, splitPoint
            self.root.varImportances[column] = self.root.varImportances[column] + \
                                            (1-performance)*len(self.rowIndex)/self.root.numRows
            L_ID, R_ID = self.splitIDs(self.rowIndex)
            self.leftChild = Node(self.root, L_ID)
            self.rightChild = Node(self.root, R_ID)
        
    
    def evaluateSplits(self):
        numTry = self.root.numTry
        
        ## evaluate numTry random subspaces
        colIndices = np.arange(self.root.numColumns)
        np.random.shuffle(colIndices)
        colIndices = colIndices[:numTry]
        
        performances, splitPoints = np.zeros(numTry), np.zeros(numTry)
        for i, index in enumerate(list(colIndices)):
            performance, splitPoint = self.split(index)
            performances[i] = performance
            splitPoints[i] = splitPoint
        
        ## check if no best split
        performances[np.isnan(performances)] = self.noPerformanceMarker
        bestIndex = np.argmax(performances)
        bestPerformance = performances[bestIndex]
        if bestPerformance == self.noPerformanceMarker:
            return None, None, None
        else:
            bestSplitPoint = splitPoints[bestIndex]
            bestColumn = colIndices[bestIndex]
            return bestColumn, bestSplitPoint, bestPerformance
        
        
    def split(self, colIndex):
        if self.root.columnTypes[colIndex] is np.dtype(np.bool):
            return self.splitBinary(colIndex)
        else:
            return self.splitContinuous(colIndex)
    
    
    
    ## sorts a continuous predictor in ascending order for the purpose of linear search for split points
    def returnSorted(self, colIndex):
        ## only to be used for continuous features
        subData = self.root.data[self.rowIndex,colIndex]
        sortIndex = np.argsort(subData)
        subData = subData[sortIndex]
        intervention = self.root.intervention[self.rowIndex][sortIndex]
        outcomes = self.root.outcomes[self.rowIndex,...][sortIndex,...]
        return subData, intervention, outcomes

    def minSize(self, L_model, R_model):
        return np.min([L_model.totalC, L_model.totalT, R_model.totalC, R_model.totalT])
    
    def splitBinary(self, colIndex):
        subData = self.root.data[self.rowIndex,colIndex]
        intervention = self.root.intervention[self.rowIndex]
        outcomes = self.root.outcomes[self.rowIndex]
        
        L_select = subData == 1
        R_select = np.logical_not(L_select)
        performance = self.noPerformanceMarker
        splitPoint = 0.5  ## will always separate 0/1 and functions the same as mode imputation
        
        L_intervention = intervention[L_select]
        R_intervention = intervention[R_select]
        L_outcomes = outcomes[L_select,...]
        R_outcomes = outcomes[R_select,...]
        
        L_model = SurvStats(L_outcomes, L_intervention)
        R_model = SurvStats(R_outcomes, R_intervention)
        
        performance = self.noPerformanceMarker
        ## make sure subgroups are of min size
        if self.minSize(L_model, R_model) > self.root.minGroup:
            potentialPerformance = (abs(L_model.getStatistic()) + abs(R_model.getStatistic()))/2
            # make sure performance is good enough
            #if stats.norm.sf(potentialPerformance) <= self.root.pval:
            performance = potentialPerformance*(self.maximize*2 - 1)
                
        return performance, splitPoint

    ## assumes pre-sorted
    def splitContinuous(self, colIndex):
        subData, intervention, outcomes = self.returnSorted(colIndex)
        reg = self.root.minGroup*2
        
        ## init all vars with results from first split check -1 (corrected in first loop iter)
        L_intervention = intervention[:reg-1]
        R_intervention = intervention[reg-1:]
        L_outcomes = outcomes[:reg-1,...]
        R_outcomes = outcomes[reg-1:,...]

        L_model = SurvStats(L_outcomes, L_intervention, outcomes[:,1])
        R_model = SurvStats(R_outcomes, R_intervention, outcomes[:,1])

        bestPerformance = self.noPerformanceMarker
        bestSplit = 0
        ## insert/delete people from groups and update vars accordingly
        for i in range(reg-1, len(subData)-reg):
            a, y, t = intervention[i], outcomes[i,0], outcomes[i,1]
            R_model.remove(y, t, a)
            L_model.insert(y, t, a)
                                
            if subData[i] != subData[i-1]:
                if self.minSize(L_model, R_model) > self.root.minGroup:
                    currentPerformance = (abs(L_model.getStatistic()) + abs(R_model.getStatistic()))/2
                    currentSplit = subData[i]
                    currentPerformance = currentPerformance*(2*self.maximize - 1)
                    if currentPerformance > bestPerformance:
                    #and stats.norm.sf(currentPerformance) <= self.root.pval:
                        bestPerformance = currentPerformance
                        bestSplit = currentSplit
                        
        return bestPerformance, bestSplit
