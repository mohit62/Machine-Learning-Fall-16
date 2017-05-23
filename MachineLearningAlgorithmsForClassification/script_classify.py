from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
from scipy import stats

import dataloader as dtl
import classalgorithms as algs
 
 
def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def geterror(ytest, predictions):
    return (100.0-getaccuracy(ytest, predictions))

 
if __name__ == '__main__':
    trainsize = 5000
    testsize = 5000
    numruns = 10
    LR=[]
    classalgs = {'RBF': algs.RadialBasisNetwork(),
                 'Logistic Regression': algs.LogitReg(),
                 'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
                 'Neural Network': algs.NeuralNet({'epochs': 100}),			 
                 }
    numalgs = len(classalgs)    

    parameters = (
        {'Beta': 1, 'nh': 4},
        {'Beta': 0.1},
        {'Beta':0.01},  )
    numparams = len(parameters)
        
    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams,numruns))
    for r in range(numruns):	
        trainset, testset = dtl.load_susy(trainsize,testsize)
        #trainset, testset = dtl.load_susy_complete(trainsize,testsize)

        print('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r)

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in classalgs.iteritems():
                # Reset learner for new parameters
                learner.reset(params)
    	    	print 'Running learner = ' + learnername + ' on parameters ' + str(learner.getparams())
    	    	# Train model
    	    	learner.learn(trainset[0], trainset[1])
    	    	# Test model
    	    	predictions = learner.predict(testset[0])
    	    	error = geterror(testset[1], predictions)
    	    	print 'Error for ' + learnername + ': ' + str(error)
                errors[learnername][p,r] = error
                

    for learnername, learner in classalgs.iteritems():
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters        
        learner.reset(parameters[bestparams])
    	print 'Best parameters for ' + learnername + ': ' + str(learner.getparams())
        Avg=1.96*np.std(errors[learnername][bestparams,:])/math.sqrt(numruns)
        print 'Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(Avg)
    LR=errors['Logistic Regression'][0]
    NB=errors ['Naive Bayes'][0]
    NN=errors['Neural Network'][0]
    print LR,NB,NN
    print "the t statistic and p value for Null hypothesis while comparing Logistic Rgression AND Naive Bayes",  stats.ttest_ind(LR,NB)
    print "the t statistic and p value for Null hypothesis while comparing Logistic Rgression and Neural Network",  stats.ttest_ind(LR,NN)
    print "the t statistic and p value for Null hypothesis while comparing Logistic Rgression and Naive Bayes",  stats.ttest_ind(NN,NB)
		