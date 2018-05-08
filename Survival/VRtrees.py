# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 22:02:06 2018

@author: adityabiswas
"""


import numpy as np
from joblib import Parallel, delayed
import multiprocessing

from copy import deepcopy as copy

from scipy import stats

def getRMST(times, n, d, c, tau):
    select = np.logical_and(n > 0, times <= tau)
    times, n, d, c = times[select], n[select], d[select], c[select]
    
    # check if anybody died before tau in this population
    if np.any(select):
        w = (n-c)/n
        S = np.cumprod(w*(1 - d/n))
        deltas = times[1:] - times[:-1]
        # add first and last rectangles separately
        area = np.sum(S[:-1]*deltas) + times[0] + S[-1]*(tau - times[-1])
    else:
        area = tau
    return area

class SurvStats(object):
    def __init__(self, outcomes, intervention, weights = None):
        super().__init__()
        if weights is None:
            weights = np.ones(len(outcomes))
        sorter = np.lexsort((1-outcomes[:,0], outcomes[:,1]))
        outcomes, intervention, weights = outcomes[sorter,...], intervention[sorter], weights[sorter]
        T_outcomes, T_weights = outcomes[intervention,...], weights[intervention]
        C_outcomes, C_weights = outcomes[np.logical_not(intervention)], weights[np.logical_not(intervention)]
        
        self.m_t, self.m_c = np.sum(T_weights), np.sum(C_weights)

        # statistics to track population
        self.times = np.unique(outcomes[:,1])
        m = len(self.times)
        self.nt, self.nc = np.zeros(m, dtype = np.float32), np.zeros(m, dtype = np.float32)
        self.dt, self.dc = np.zeros(m, dtype = np.float32), np.zeros(m, dtype = np.float32)
        self.ct, self.cc = np.zeros(m, dtype = np.float32), np.zeros(m, dtype = np.float32)
                
        for i in range(m):
            T_index = np.where(T_outcomes[:,1] == self.times[i])[0]
            self.dt[i] = np.sum(T_outcomes[T_index,0]*T_weights[T_index])
            self.ct[i] = np.sum((1-T_outcomes[T_index,0])*T_weights[T_index])
            select = T_outcomes[:,1] >= self.times[i]
            self.nt[i] = np.sum(T_weights[select])
            
            C_index = np.where(C_outcomes[:,1] == self.times[i])[0]
            self.cc[i] = np.sum((1-C_outcomes[C_index,0])*C_weights[C_index])
            self.dc[i] = np.sum(C_outcomes[C_index,0]*C_weights[C_index])
            select = C_outcomes[:,1] >= self.times[i]
            self.nc[i] = np.sum(C_weights[select])
            
            
        if len(T_outcomes) > 0:
            self.deathsT, self.totalT = self.getTotal(T_outcomes)
        else:
            self.deathsT, self.totalT = 0, 0
        if len(C_outcomes) > 0:
            self.deathsC, self.totalC = self.getTotal(C_outcomes)
        else:
            self.deathsC, self.totalC = 0, 0


    def getTotal(self, outcomes):

        deathBool = outcomes[:,0] == 1
        numDeaths = np.sum(deathBool)
        if numDeaths == 0:
            boundaryTime = outcomes[-1,1]
        else:
            boundaryTime = np.mean(outcomes[deathBool,1])
        #avgDeathTime = np.minimum(avgDeathTime, 365.25*4)
        numEffectiveAlive = np.sum(np.minimum(outcomes[np.logical_not(deathBool),1]/boundaryTime, 1))
        return numDeaths, numDeaths + numEffectiveAlive
    
        
        
    def getN(self):
        return self.nt + self.nc
    def getD(self):
        return self.dt + self.dc
    def getTimes(self):
        return self.times
    
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


    def getRMSTdiff(self, tau):
        areaT = getRMST(self.times, self.nt, self.dt, self.ct, tau = tau)
        areaC = getRMST(self.times, self.nc, self.dc, self.cc, tau = tau)
        return areaT - areaC
    
        


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
        self.intervention, self.outcomes = intervention, outcomes
        self.numRows, self.numColumns = np.shape(data)
        self.recordColumnTypes(data)
        if colNames is not None:
            self.colNames = colNames
        
        ## run training over all trees
        if self.verbose:
            verbose = 12
        else:
            verbose = 0
            
        ncores = multiprocessing.cpu_count() - 2
        self.trees = Parallel(n_jobs = ncores, backend = 'multiprocessing', verbose = verbose)(delayed(self.makeTree)(data, intervention, 
                              outcomes) for i in range(self.numTrees))
        self.fitted = True
        
#    def getWeights(self, data, treeNum):
#        results = self.trees[treeNum].predict(data)
#        return results
#    
#    def getITE(self, weights, index):
#        select = weights[index,:] > 0
#        model = SurvStats(self.outcomes[select,:], self.intervention[select], weights[index, select])
#        return model.getRMSTdiff(tau = 7)
#
#    def predict(self, data):
#        assert self.fitted
#        ncores = multiprocessing.cpu_count() - 2
#        weights = Parallel(n_jobs = ncores, backend = 'multiprocessing', verbose = 12)(delayed(self.getWeights)(data, i) 
#                                            for i in range(self.numTrees))
#        weights = np.stack(weights, axis = 2)
#        weights = np.sum(weights, axis = 2)/self.numTrees
#        
#        numRows = len(data)
#        ITE = Parallel(n_jobs = ncores, backend = 'multiprocessing', verbose = 12)(delayed(self.getITE)(weights, i) 
#                                            for i in range(numRows))
#        return np.array(ITE)
    
    
    def predict(self, data):
        assert self.fitted
        numRows = len(data)
        allResults = np.zeros((numRows, self.numTrees))
        
        for i in range(self.numTrees):
            results, totals = self.trees[i].predict(data)
            allResults[:,i] = results

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
        
#    def predict(self, data):
#        self.data = data
#        self.ID = np.arange(len(data))
#        self.results = np.zeros((len(data), self.numRows), dtype = np.bool)
#        self.tree.predict(self.ID)
#        return self.results
        
    
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
        
#    def predict(self, rowIndex):
#        if self.isLeaf:
#            for i in range(len(rowIndex)):
#                self.root.results[rowIndex[i],self.leafValue] = True
#        else:
#            L_ID, R_ID = self.splitIDs(rowIndex)
#            self.leftChild.predict(L_ID)
#            self.rightChild.predict(R_ID)
            
            
    def countLeaf(self):
        if self.isLeaf:
            self.root.numLeaves += 1
        else:
            self.leftChild.countLeaf()
            self.rightChild.countLeaf()
    
    def splitIDs(self, rowIndex):
        data, ID = self.root.data[rowIndex,self.column], self.root.ID[rowIndex]
        data = self.impute(copy(data), self.column)
        L_select = data <= self.splitPoint
        R_select = np.logical_not(L_select)
        L_ID, R_ID = ID[L_select], ID[R_select]
        return L_ID, R_ID
    
    def makeLeaf(self):
        self.isLeaf = True
        outcomes = self.root.outcomes[self.rowIndex,...]
        intervention = self.root.intervention[self.rowIndex]
        model = SurvStats(outcomes, intervention)
#        self.leafValue = self.rowIndex
        self.leafValue = model.getRMSTdiff(7)
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
            #numTry = 5
            colIndices = np.arange(self.root.numColumns)
            #colIndices = np.random.permutation(colIndices)[:numTry]
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
        subData = self.impute(copy(self.root.data[self.rowIndex,colIndex]), colIndex)
        sortIndex = np.argsort(subData)
        subData = subData[sortIndex]
        intervention = self.root.intervention[self.rowIndex][sortIndex]
        outcomes = self.root.outcomes[self.rowIndex,...][sortIndex,...]
        return subData, intervention, outcomes

    def minSize(self, L_model, R_model):
        return np.min([L_model.deathsC, L_model.deathsT, R_model.deathsC, R_model.deathsT])
    
    def impute(self, data, colIndex):
        needsImpute = np.isnan(data)
        finiteVals = data[np.logical_not(needsImpute)]
        if self.root.columnTypes[colIndex] == 0:
            val = stats.mode(finiteVals)[0][0]
        else:
            val = np.mean(finiteVals)
        data[needsImpute] = val
        return data
    
    def evaluate(self, L_model, R_model):
        L_m, R_m = (L_model.totalC + L_model.totalT), (R_model.totalC + R_model.totalT)
        m = L_m + R_m
        L_area = L_model.getRMSTdiff(tau = 7)*L_m/m
        R_area = R_model.getRMSTdiff(tau = 7)*R_m/m
        return abs(L_area) + abs(R_area)
        
    
    def splitBinary(self, colIndex):
        subData = self.impute(copy(self.root.data[self.rowIndex,colIndex]), colIndex)
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
                    performance = self.evaluate(L_model, R_model)
                    # make sure performance is good enough
                    #if stats.norm.sf(potentialPerformance) <= self.root.pval:
                    performance *= (self.maximize*2 - 1)
                
        return performance, splitPoint

    ## assumes pre-sorted
    def splitContinuous(self, colIndex):
        reg = self.root.minGroup*2
        subData, intervention, outcomes = self.returnSorted(colIndex)
        split_a = np.random.uniform(subData[reg], subData[-reg])
        split_b = np.random.uniform(subData[reg], subData[-reg])
        split = (split_a + split_b)/2
    
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
                performance = self.evaluate(L_model, R_model)
                performance = performance*(2*self.maximize - 1)
                            
        return performance, split
