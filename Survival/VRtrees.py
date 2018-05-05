# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 22:02:06 2018

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
    def __init__(self, outcomes, intervention):
        super().__init__()
        sorter = np.lexsort((1-outcomes[:,0], outcomes[:,1]))
        outcomes, intervention = outcomes[sorter,...], intervention[sorter]
        T_outcomes = outcomes[intervention,...]
        C_outcomes = outcomes[np.logical_not(intervention)]

        # statistics to track population
        self.times = np.unique(outcomes[outcomes[:,0] == 1,1])
        m = len(self.times)
        self.nt, self.nc = np.zeros(m, dtype = np.int64), np.zeros(m, dtype = np.int64)
        self.dt, self.dc = np.zeros(m, dtype = np.int64), np.zeros(m, dtype = np.int64)
                
        # initialize with current data
        for i in range(m):
            T_index = np.where(T_outcomes[:,1] == self.times[i])[0]
            self.dt[i] = np.sum(T_outcomes[T_index,0])
            self.nt[i] = np.sum(T_outcomes[:,1] >= self.times[i])
            C_index = np.where(C_outcomes[:,1] == self.times[i])[0]
            self.dc[i] = np.sum(C_outcomes[C_index,0])
            self.nc[i] = np.sum(C_outcomes[:,1] >= self.times[i])
          
        
        self.totalT = self.getTotal(T_outcomes)
        self.totalC = self.getTotal(C_outcomes)


    def getTotal(self, outcomes):

        deathBool = outcomes[:,0] == 1
        numDeaths = np.sum(deathBool)
        if numDeaths == 0:
            avgDeathTime = outcomes[-1,1]
        else:
            avgDeathTime = np.mean(outcomes[deathBool,1])
        #avgDeathTime = np.minimum(avgDeathTime, 365.25*4)
        numEffectiveAlive = np.sum(np.minimum(outcomes[np.logical_not(deathBool),1]/avgDeathTime, 1))
        return numDeaths + numEffectiveAlive
        
        
    
    def getStatistic(self):
        n = self.nc + self.nt
        d = self.dc + self.dt
        select = np.logical_and(n-d > 0, np.logical_and(self.nt > 0, self.nc > 0))
        
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
                print(nt)
                print(nc)
            return val
        else:
            return 0
    
        
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
    def __init__(self, numTrees, minGroup, alpha = 0.5, verbose = False):
        super().__init__()
        self.numTrees = numTrees
        self.minGroup = minGroup
        self.alpha = alpha
        self.verbose = verbose
        self.fitted = False
        self.counter = 0
        
    def makeTree(self, data, intervention, outcomes):
        np.random.seed()
        tree = Root(data, intervention, outcomes, self.minGroup, self.alpha,
                    self.columnTypes, self.colNames)
        self.counter += 1
        return tree

    def fit(self, data, intervention, outcomes, colNames = None):
        ## data is predictor float matrix
        ## intervention is boolean array
        ## outcomes is matrix with Y in col 0 and T in col 1
        numRows, self.numColumns = np.shape(data)
        self.recordColumnTypes(data)
        if colNames is not None:
            self.colNames = colNames
        
        ## run training over all trees
        if self.verbose:
            verbose = 12
        else:
            verbose = 0
        self.trees = Parallel(n_jobs = -1, backend = 'multiprocessing', verbose = verbose)(delayed(self.makeTree)(data, intervention, 
                              outcomes) for i in range(self.numTrees))
        self.fitted = True
        
    def predict(self, data):
        assert self.fitted
        numRows = len(data)
        allResults = np.zeros((numRows, self.numTrees))
#        weights = np.zeros((numRows, self.numTrees))
#        
        for i in range(self.numTrees):
            results, totals = self.trees[i].predict(data)
            allResults[:,i] = results
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
        # 1 for float, 0 for bool
        self.columnTypes = [1 for i in range(self.numColumns)]
        for i in range(self.numColumns):
            ## cutoff of 3 used because mean imputation is used creating a 3rd unique value for boolean vars
            if len(np.unique(data[:200,i])) <= 10 :
                self.columnTypes[i] = 0

            

class Root(object):
    def __init__(self,  data, intervention, outcomes, minGroup, alpha, columnTypes, colNames):
        super().__init__()
        self.numRows, self.numColumns = np.shape(data)
        self.ID = np.arange(self.numRows)
        self.data = data
        self.intervention = intervention
        self.outcomes = outcomes
        self.columnTypes = columnTypes
        self.colNames = colNames
        self.minGroup = minGroup
        self.alpha = alpha
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

        self.leafValue = model.getRMSTdiff()
        self.leafTotal = model.totalC + model.totalT
        
    
    def fit(self):
        column, splitPoint, performance = self.evaluateSplits()
        if column is None:
            self.makeLeaf()
        else:
            self.column, self.splitPoint = column, splitPoint
            self.root.varImportances[column] = self.root.varImportances[column] + \
                                            (performance)*len(self.rowIndex)/self.root.numRows
            L_ID, R_ID = self.splitIDs(self.rowIndex)
            self.leftChild = Node(self.root, L_ID)
            self.rightChild = Node(self.root, R_ID)
        
    
    def evaluateSplits(self):
        if len(self.rowIndex) >= self.root.minGroup*4 + 1:
            
            ## evaluate numTry random subspaces
            colIndices = np.arange(self.root.numColumns)
            if np.random.uniform() < self.root.alpha:
                numTry = self.root.numColumns
            
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
            else:
                index = np.random.choice(colIndices)
                performance, splitPoint = self.split(index)
                if performance == self.noPerformanceMarker:
                    return None, None, None
                else:
                    return index, splitPoint, performance

        else:
            return None, None, None
        
        
    def split(self, colIndex):
        if self.root.columnTypes[colIndex] == 0:
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
    
        uniqueVals = np.unique(subData)
        splitPoint = 0
        performance = self.noPerformanceMarker
        if len(uniqueVals) > 1:
            uniqueVals = uniqueVals[:-1]
            splitPoint = np.random.choice(uniqueVals)
            
            L_select = subData <= splitPoint
            R_select = np.logical_not(L_select)
            
            L_intervention = intervention[L_select]
            R_intervention = intervention[R_select]
            
            reg = self.root.minGroup*2
            if len(L_intervention) > reg and len(R_intervention) > reg:
                L_outcomes = outcomes[L_select,...]
                R_outcomes = outcomes[R_select,...]
                
                L_model = SurvStats(L_outcomes, L_intervention)
                R_model = SurvStats(R_outcomes, R_intervention)
            
                ## make sure subgroups are of min size
                if self.minSize(L_model, R_model) > self.root.minGroup:
                    m = len(self.rowIndex)
                    performance = abs(L_model.getStatistic())*(L_model.totalC + L_model.totalT)/m
                    performance += abs(R_model.getStatistic())*(R_model.totalC + R_model.totalT)/m
                    # make sure performance is good enough
                    #if stats.norm.sf(potentialPerformance) <= self.root.pval:
                    performance *= (self.maximize*2 - 1)
                
        return performance, splitPoint

    ## assumes pre-sorted
    def splitContinuous(self, colIndex):
        reg = self.root.minGroup*2
        subData, intervention, outcomes = self.returnSorted(colIndex)
        split = np.random.uniform(subData[reg], subData[-reg])
        #split_b = np.random.uniform(subData[reg], subData[-reg])
        #split = (split_a + split_b)/2
    
        L_split = subData <= split
        R_split = np.logical_not(L_split)

        L_intervention = intervention[L_split]
        R_intervention = intervention[R_split]
        L_outcomes = outcomes[L_split,...]
        R_outcomes = outcomes[R_split,...]
        
        
        performance = self.noPerformanceMarker
        if len(L_intervention) > reg and len(R_intervention) > reg:

            L_model = SurvStats(L_outcomes, L_intervention)
            R_model = SurvStats(R_outcomes, R_intervention)
        
            if self.minSize(L_model, R_model) > self.root.minGroup:
                m = len(self.rowIndex)
                performance = abs(L_model.getStatistic())*(L_model.totalC + L_model.totalT)/m
                performance += abs(R_model.getStatistic())*(R_model.totalC + R_model.totalT)/m
                performance = performance*(2*self.maximize - 1)
                            
        return performance, split
