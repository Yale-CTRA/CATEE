#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 09:50:10 2018

@author: aditya
"""


import numpy as np
from joblib import Parallel, delayed
import time




class RMST(object):
    # 3 scenarios: neither (0), branch w/ future removals (1) branch w/ future insertions (2)
    # removeData should be set for scenario 1 and outcomes should be all; allTimes should be scenario 2
    def __init__(self, outcomes, scenario = 0, removeData = None, allTimes = None, tau = 365.25*4):
        super().__init__()
        self.tau = tau
        sorter = np.lexsort((1-outcomes[:,0], outcomes[:,1]))
        outcomes = outcomes[sorter,:]
        
        if scenario == 0:    
            self.n = np.arange(len(outcomes))[::-1] + 1
            eventBool = outcomes[:,0] == 1
            numEvents = np.sum(eventBool)
            eventTimes = outcomes[eventBool,1]
            self.times, tIndex, self.d = np.unique(eventTimes, return_index = True, return_counts = True)
            self.n = self.n[eventBool][tIndex]
            self.total = numEvents + np.sum(np.minimum(outcomes[np.logical_not(eventBool),1], self.tau)/self.tau)

        elif scenario == 1:
            self.n = np.arange(len(outcomes))[::-1] + 1
            self.times, tIndex = np.unique(outcomes[:,1], return_index = True)
            self.n = self.n[tIndex]
            
            m = len(self.n)
            self.d = np.zeros(m, dtype = np.int32)
            for i in range(m-1):
                self.d[i] = np.sum(outcomes[tIndex[i]:tIndex[i+1],0])
            self.d[-1] = np.sum(outcomes[tIndex[-1]:,0])
            
            eventBool = outcomes[:,0] == 1
            numEvents = np.sum(eventBool)
            self.total = numEvents + np.sum(np.minimum(outcomes[np.logical_not(eventBool),1], self.tau)/self.tau)
            
            for i in range(len(removeData)):
                self.remove(removeData[i,0], removeData[i,1])
            
        else: # scenario 2: allTimes must be set for this scenario
            self.times = np.unique(allTimes)
                
            # statistics to track population
            m = len(self.times)
            self.n = np.zeros(m)
            self.d = np.zeros(m)
            self.total = 0
            
            # initialize with current data
            counter = 0
            i = 0
            while counter < len(outcomes):
                if self.times[i] == outcomes[counter,1]:
                    self.n[:i+1] = self.n[:i+1] + 1
                    if outcomes[counter,0] == 1:
                        self.d[i] = self.d[i] + 1
                        self.total += 1
                    else:
                        self.total += min(outcomes[counter, 1]/self.tau, 1)
                    # if match: shift index in outcomes
                    counter += 1
                else:
                    # if no match: shift index in potential times
                    i += 1
                
    def insert(self, y, t):
#        self.total += 1
        index = self.where(t)
        self.n[:index+1] = self.n[:index+1] + 1
        if y == 1:
            self.d[index] = self.d[index] + 1
            self.total += 1
        else:
            self.total += min(t/self.tau, 1)
            
    def remove(self, y, t):
#        self.total -= 1
        index = self.where(t)
        self.n[:index+1] = self.n[:index+1] - 1
        if y == 1:
            self.d[index] = self.d[index] - 1
            self.total -= 1
        else:
            self.total -= min(t/self.tau, 1)
        
    def where(self, t):
        index = np.where(self.times == t)[0][0]
        return index       
    

    def performance(self):
        select = np.logical_and(self.d > 0, self.times < self.tau)
        
        # check if anybody died before tau in this population
        if np.any(select):
            times, n, d = self.times[select], self.n[select], self.d[select]   
            S = np.cumprod(1 - d/n)
            deltas = times[1:] - times[:-1]
            # add first and last rectangles separately
            area = np.sum(S[:-1]*deltas) + times[0] + S[-1]*(self.tau - times[-1])
            return area

        else:
            return self.tau
        


class RandomForest(object):
    def __init__(self, numTrees, numTry, bootstrapPercent, minGroup, minPerformance, verbose = True):
        super().__init__()
        self.numTrees = numTrees
        self.numTry = numTry
        self.bootstrapPercent = bootstrapPercent
        self.minGroup = minGroup
        self.minPerformance = minPerformance
        self.verbose = verbose
        self.fitted = False
        self.counter = 0
        
    def makeTree(self, data, intervention, outcomes, bootstrapCut):
        np.random.seed()
        sorter = np.random.permutation(np.arange(len(data)))
        data, intervention, outcomes = data[sorter,...], intervention[sorter], outcomes[sorter,...]
        tree = Root(data[:bootstrapCut,...], intervention[:bootstrapCut], 
                    outcomes[:bootstrapCut,...], self.numTry, self.minGroup, self.minPerformance, self.columnTypes)
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
        for i in range(self.numTrees):
            allResults[:,i] = self.trees[i].predict(data)
        #return np.mean(allResults, axis = 1)
        return allResults
    
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
    def __init__(self,  data, intervention, outcomes, numTry, minGroup, minPerformance, columnTypes):
        super().__init__()
        self.numRows, self.numColumns = np.shape(data)
        self.ID = np.arange(self.numRows)
        self.data = data
        self.intervention = intervention
        self.outcomes = outcomes
        self.columnTypes = columnTypes
        self.numTry = numTry
        self.minGroup = minGroup
        self.minPerformance = minPerformance
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
        self.tree.predict(self.ID)
        ## free memory
        self.data, self.numRows, self.ID = None, None, None
        return self.results

class Node(object):
    def __init__(self, root, rowIndex):
        super().__init__()
        self.root = root
        self.rowIndex = rowIndex
        ## peformance variable at this value indicates no appropriate split was found
        self.noPerformanceMarker = 0
        
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
        else:
            L_ID, R_ID = self.splitIDs(rowIndex)
            self.leftChild.predict(L_ID)
            self.rightChild.predict(R_ID)
    
    
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
        outcomesTreated = outcomes[intervention,...]
        outcomesControl = outcomes[np.logical_not(intervention),...]
        T_model = RMST(outcomesTreated)
        C_model = RMST(outcomesControl)

        self.leafValue = T_model.performance() - C_model.performance()
        
    
    def fit(self):
        column, splitPoint, performance = self.evaluateSplits()
        if column is None:
            self.makeLeaf()
        else:
            self.column, self.splitPoint = column, splitPoint
            self.root.varImportances[column] = self.root.varImportances[column] + \
                                            performance*len(self.rowIndex)/self.root.numRows
            L_ID, R_ID = self.splitIDs(self.rowIndex)
            self.leftChild = Node(self.root, L_ID)
            self.rightChild = Node(self.root, R_ID)
        
    
    def evaluateSplits(self):
        ## returning None, None is indication that no appropriate split was found/exists
        if len(self.rowIndex)/4 < self.root.minGroup:
            return None, None, None
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
        performances[np.isnan(performances)] = -100
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
        LT_model = RMST(L_outcomes[L_intervention,...])
        LC_model = RMST(L_outcomes[np.logical_not(L_intervention),...])
        RT_model = RMST(R_outcomes[R_intervention,...])
        RC_model = RMST(R_outcomes[np.logical_not(R_intervention),...])
        if np.min([LT_model.total, LC_model.total, RT_model.total, RC_model.total]) >= self.root.minGroup:
            L_weight, R_weight = self.getWeights(LT_model, LC_model, RT_model, RC_model)
            performance = L_weight*np.abs(LT_model.performance() - LC_model.performance()) + \
                                R_weight*np.abs(RT_model.performance() - RC_model.performance())
            weight = (LT_model.total + LC_model.total + RT_model.total + RC_model.total)/self.root.n
            if performance*weight < self.root.minPerformance:
                performance = self.noPerformanceMarker
        
        return performance, splitPoint
            
    def getWeights(self, LT_model, LC_model, RT_model, RC_model):
        L_total, R_total = LT_model.total + LC_model.total, RT_model.total + RC_model.total
        total = L_total + R_total
        return L_total/total, R_total/total
        

    ## assumes pre-sorted
    def splitContinuous(self, colIndex):
        subData, intervention, outcomes = self.returnSorted(colIndex)
        reg = 2*self.root.minGroup
        
        ## init all vars with results from first split check
        T_allTimes = outcomes[intervention, 1]
        C_allTimes = outcomes[np.logical_not(intervention), 1]
        L_intervention = intervention[:reg]
       # R_intervention = intervention[reg:]
        L_outcomes = outcomes[:reg,...]
        #R_outcomes = outcomes[reg:,...]
        LT_model = RMST(L_outcomes[L_intervention,...], scenario = 2, allTimes = T_allTimes)
        LC_model = RMST(L_outcomes[np.logical_not(L_intervention),...], scenario = 2, allTimes = C_allTimes)
        RT_model = RMST(outcomes[intervention,...], scenario = 1, removeData = L_outcomes[L_intervention,...])
        RC_model = RMST(outcomes[np.logical_not(intervention),...], scenario = 1, removeData = L_outcomes[np.logical_not(L_intervention),...])
        weight = (LT_model.total + LC_model.total + RT_model.total + RC_model.total)/self.root.n

        
        bestPerformance = self.noPerformanceMarker
        bestSplit = 0
        if np.min([LT_model.total, LC_model.total, RT_model.total, RC_model.total]) >= self.root.minGroup:
            L_weight, R_weight = self.getWeights(LT_model, LC_model, RT_model, RC_model)
            bestPerformance = L_weight*np.abs(LT_model.performance() - LC_model.performance()) + \
                                R_weight*np.abs(RT_model.performance() - RC_model.performance())
            if bestPerformance*weight < self.root.minPerformance:
                 bestPerformance = self.noPerformanceMarker
            else:
                bestSplit = subData[reg]
        
        ## insert/delete people from groups and update vars accordingly
        for i in range(reg, len(subData)-reg):
            a, y, t = intervention[i], outcomes[i,0], outcomes[i, 1]
            if a:
                RT_model.remove(y, t)
                LT_model.insert(y, t)
            else:
                RC_model.remove(y, t)
                LC_model.insert(y, t)
                                
            if subData[i] != subData[i+1]:
                if np.min([LT_model.total, LC_model.total, RT_model.total, RC_model.total]) >= self.root.minGroup:
                    L_weight, R_weight = self.getWeights(LT_model, LC_model, RT_model, RC_model)
                    currentPerformance = L_weight*np.abs(LT_model.performance() - LC_model.performance()) + \
                                        R_weight*np.abs(RT_model.performance() - RC_model.performance())
                    currentSplit = subData[i]
                    if currentPerformance > bestPerformance and currentPerformance*weight >= self.root.minPerformance:
                        bestPerformance = currentPerformance
                        bestSplit = currentSplit
        return bestPerformance, bestSplit
