# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 12:25:26 2018

@author: adityabiswas
"""

import sys
import numpy as np
import time
import pandas as pd
sys.path.append('C:\\Users\\adityabiswas\\Documents\\ML Projects\\SPRINT Code\\third')

from evaluate_main import retrieveData
from VRtrees import RandomForest as RF
import evaluate_func as E

def ID2str(IDs):
    return np.array(['S' + '0'*(5-len(string)) + string for string in IDs.astype(np.int32).astype(np.unicode_).tolist()])

def getHrsMins(time):
    hours = int(np.floor(time))
    minutes = np.round(60*(time-hours), decimals = 1)
    text = str(hours) + ' hrs ' + str(minutes) + ' mins'
    return text
    

def main():
    data = retrieveData()
    
    
    IDs = np.sort(np.concatenate([data.train['id'], data.test['id']]))
    IDs = ID2str(IDs)
    allResults = pd.DataFrame(index = IDs)
    
    
    totalTime = 0
    iterations = 50
    auucs = np.zeros(iterations)
    varImportances = pd.DataFrame(index = data.infoDict['x'])
    badSeeds = [37, 13, 25, 26, 34]
    ## associated with [-640, -450, -450, -400, -425]
    for i in range(iterations):
        data.refresh(seed = i)
        
        Xtrain = data.train['x']
        Atrain = data.train['a'] == 1
        Otrain = np.stack([data.train['y'][:,0], data.train['t'][:,0]], axis = 1)
        
        ## fit
        numTrees = 1000
        start = time.time()
        model = RF(numTrees, minGroup = 80, alpha = 0, verbose = False)
        model.fit(Xtrain, Atrain, Otrain, colNames = data.infoDict['x'])
        stop = time.time()
        totalTime += (stop-start)/(60*60)
        avgTime = totalTime/(i+1)
        timeLeft = avgTime*(iterations - (i+1))
        print('Iteration: ', i, ' | Time Completed: ', getHrsMins(totalTime),
              ' | Time Left: ', getHrsMins(timeLeft))
        
    
        ## evaluate
        dataEval = data.test
        results = model.predict(dataEval['x'])
        avgLeaves = model.getNumLeaves()
        Y, T, A = E.setVars(dataEval)
        #results = np.mean(results, axis = 1)
        auuc = E.recLogRank(results, Y[:,0], T[:,0], A)
        
        evalIDs = ID2str(dataEval['id'])
        results = pd.Series(results, index = evalIDs)
        allResults[i+1] = results
        auucs[i] = auuc
        print('Performance: ', np.round(auuc, decimals = 2), ' | Running avg: ', 
              np.round(np.mean(auucs[:i+1]), decimals = 2), ' | Avg Leaves: ', avgLeaves, '\n')
        
        ## var Importances
        varImportances[i] = model.getVarImportances()
    
    ## after all iterations have run
    auucs = pd.Series(auucs)
    folder = 'C:\\Users\\adityabiswas\\Documents\\ML Projects\\SPRINT Code\\third\\'
    name = 'LogRank5'
    auucs.to_csv(folder + 'auucs' + name + '.csv')
    allResults.to_csv(folder + 'results' + name + '.csv')
    varImportances.to_csv(folder + 'importances' + name + '.csv')

if __name__ == "__main__":
    main()