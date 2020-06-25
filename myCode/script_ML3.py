import numpy as np
import random as random
import scipy.stats as stats
import scipy.io as spio
import os
import itertools
import csv

#import classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

# import miscellaneous tools
from sklearn.metrics import zero_one_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix

# load labels
LBs = '/Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/myData/fisherVectors/numClusters10_exemplarSize3/LB.mat'
labels = spio.loadmat(LBs)['LB'][0]

# choose fisher vectors to load
#FV_names = ('brightness', 'mfc', 'chroma', 'eng', 'keystrength', 'roughness', 'zerocross', 'combined')
#FV_names_combine = ('brightness', 'mfc', 'chroma', 'eng', 'keystrength', 'roughness', 'zerocross')

#FV_names_combine = ('brightness', 'mfc', 'chroma', 'eng', 'keystrength', 'roughness', 'zerocross')
#FV_names_combine = ('brightness', 'new_MFCC', 'binned_wavlet', 'chroma', 'eng', 'keystrength', 'roughness', 'zerocross')
#FV_names_combine = ('brightness', 'new_MFCC', 'binned_wavlet', 'chroma', 'eng', 'keystrength', 'roughness', 'zerocross', 'mfc')
FV_names_combine = ('brightness', 'new_MFCC', 'chroma', 'eng', 'keystrength', 'roughness', 'zerocross', 'mfc')
#FV_names_combine = ('mfc','new_MFCC')
#FV_names_combine = ('binned_wavlet',)
# function to make all combinations of features
def all_subsets(ss):
  return itertools.chain(*map(lambda x: itertools.combinations(ss, x), range(0, len(ss)+1)))

# initialize dictionaries for each classifer
dictRF={}
dictKNN={}
dictMNB={}
dictSVM={}
dictDT={}

# choose classifier to run
doSVM = 1
doKNN = 1
doRF = 1
doMNB = 1
doDT = 1

## SAVE DATA
doSave = 0

#baseDirFVs = ('/Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/myData/fisherVectors/numClusters10_exemplarSize3/','/Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/myData/fisherVectors/numClusters10_exemplarSize5/')
baseDirFVs = ('/Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/myData/fisherVectors/numClusters10_exemplarSize3/',)


for baseDirFV in baseDirFVs:
    print baseDirFV
    for FV_namesList in all_subsets(FV_names_combine):
        if not FV_namesList:
            continue
        #...
        if len(FV_namesList) != 8:
            continue
        # if 'new_MFCC' and 'binned_wavlet' not in FV_namesList:
        #     continue
        #...
        print 'Current FVs: ', FV_namesList
        if len(FV_namesList) == 1:
            FV_name = FV_namesList[0]
            FV_file = baseDirFV+FV_name+'.mat'

            # load into arrays
            FVs = spio.loadmat(FV_file)['FV']
            N = FVs.shape[1]
        else:
            counter = -1
            for FV_name in FV_namesList:
                counter += 1
                FV_file = baseDirFV+FV_name+'.mat'
                if counter is 0:
                    FVs = spio.loadmat(FV_file)['FV']
                else:
                    FVs_current = spio.loadmat(FV_file)['FV']
                    FVs = np.concatenate((FVs,FVs_current))
                #...
            #...

        # cross validation
        k_folds = 10
        skf = StratifiedKFold(labels,n_folds=k_folds)

        # random forest classifer learning
        maxLearners = 100
        maxDepth = 5
        avgErrRF = 0.0
        errRF =  np.zeros(k_folds)
        recallRF = np.zeros(k_folds)
        precisionRF = np.zeros(k_folds)
        CM_RF = np.zeros((10,10))

        # KNN learning
        k = 10
        avgErrKNN = 0.0
        errKNN =  np.zeros(k_folds)
        recallRF = np.zeros(k_folds)
        precisionRF = np.zeros(k_folds)
        CM_KNN = np.zeros((10,10))

        # multinomial NB learning
        avgErrMNB = 0.0
        errMNB = np.zeros(k_folds)
        recallRF = np.zeros(k_folds)
        precisionRF = np.zeros(k_folds)
        CM_MNB = np.zeros((10,10))

        # svm
        avgErrSVM = 0.0
        #errSVM = np.zeros(k_folds)
        errSVM =  np.zeros(k_folds)
        recallRF = np.zeros(k_folds)
        precisionRF = np.zeros(k_folds)
        CM_SVM = np.zeros((10,10))

        # decision tree classifier
        maxDepth = 5
        avgErrDT = 0.0
        #errDT =  np.zeros(k_folds)
        errDT =  np.zeros(k_folds)
        recallRF = np.zeros(k_folds)
        precisionRF = np.zeros(k_folds)
        CM_DT = np.zeros((10,10))    

        # cross validation
        # loop over training and testing indices
        print 'Cross validating...'
        counter = -1
        for TRi,TEi in skf:
            counter += 1
            #print 'Iteration: ', counter
            # assign training and testing data
            X_TR = FVs[:,TRi]
            X_TE = FVs[:,TEi]
            Y_TR = labels[TRi]
            Y_TE = labels[TEi]

            if doRF:
                # random forest classifier
                rf = RandomForestClassifier(n_estimators = maxLearners, max_depth = maxDepth)
                # fit the random forest classifer to the training data
                rf.fit(X_TR.T,Y_TR)
                # make prediction
                predRF = rf.predict(X_TE.T)
                # determine error of classification
                errRF[counter] = zero_one_loss(predRF,Y_TE)
                #print 'Error from Random Forest Classifier: ', errRF[counter]
                avgErrRF += (1./k_folds) * errRF[counter]
                # determine confusion matrix
                CM_RF += confusion_matrix(predRF,Y_TE)
            #...

            if doDT:
                # decison tree classifier
                dt = DecisionTreeClassifier(max_depth = maxDepth)
                dt.fit(X_TR.T,Y_TR)
                # make prediction
                predDT = dt.predict(X_TE.T)
                # determine error of classification
                errDT[counter] = zero_one_loss(predDT,Y_TE)
                #print 'Error from Decision Tree Classifier: ', errDT[counter]
                avgErrDT += (1./k_folds) * errDT[counter]
                # determine confusion matrix
                CM_DT += confusion_matrix(predDT,Y_TE)
            #...

            if doKNN:
                # KNN classifier
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_TR.T,Y_TR)
                predKNN = knn.predict(X_TE.T)
                errKNN[counter] = zero_one_loss(predKNN,Y_TE)
                #print 'Error from KNN: ', errKNN[counter]
                avgErrKNN += (1./k_folds) * errKNN[counter]
                #determine confusion matrix
                CM_KNN += confusion_matrix(predKNN,Y_TE)
            #...

            if doMNB:
                # multinomial naive bayes
                mnb = MultinomialNB()
                mnb.fit(X_TR.T-np.ones(X_TR.T.shape)*np.min(X_TR.T)+np.ones(X_TR.T.shape),Y_TR)
                predMNB = mnb.predict(X_TE.T-np.ones(X_TE.T.shape)*np.min(X_TE.T)+np.ones(X_TE.T.shape))
                errMNB[counter] = zero_one_loss(predMNB,Y_TE)
                #print 'Error from multinomial Naive Bayes: ', errMND[counter]
                avgErrMNB += (1./k_folds) * errMNB[counter]
                # determine confusion matrix   
                CM_MNB += confusion_matrix(predMNB,abs(Y_TE))
            #...

            if doSVM:
                # SVM
                #suppVM = svm.SVC(kernel='poly',degree=3,coef0=1)
                suppVM = svm.LinearSVC()
                #suppVM = svm.SVC()
                suppVM.fit(X_TR.T,Y_TR)
                predSVM = suppVM.predict(X_TE.T)
                errSVM[counter] = zero_one_loss(predSVM,Y_TE)
                #print 'Error from SVM: ', errSVM[counter]
                avgErrSVM += (1./k_folds) * errSVM[counter]
                # determine confusion matrix
                CM_SVM += confusion_matrix(predSVM,Y_TE)
            #...
        #...
        if doRF:
            print 'Average error from RF:'
            print '%4.2f%s' % (100 * avgErrRF, '%')
            print 'std. dev:'
            print '%4.2f%s' % (100 * np.std(errRF), '%')
        #...
        if doKNN:
            print 'Average error from KNN:'
            print '%4.2f%s' % (100 * avgErrKNN, '%')
            print 'std. dev:'
            print '%4.2f%s' % (100 * np.std(errKNN), '%')
        #...
        if doSVM:
            print 'Average error from SVM:'
            print '%4.2f%s' % (100 * avgErrSVM, '%')
            print 'std. dev:'
            print '%4.2f%s' % (100 * np.std(errSVM), '%')
        #...
        if doMNB:
            print 'Average error from MNB:'
            print '%4.2f%s' % (100 * avgErrMNB, '%')
            print 'std. dev:'
            print '%4.2f%s' % (100 * np.std(errMNB), '%')
        #...
        if doDT:
            print 'Average error from DT:'
            print '%4.2f%s' % (100 * avgErrDT, '%')
            print 'std. dev:'
            print '%4.2f%s' % (100 * np.std(errDT), '%')
        #...
        print ' '

        # store avg error and standard deviation of error in dictionary
        dictRF[FV_namesList]=(avgErrRF,np.std(errRF),errRF)
        dictSVM[FV_namesList]=(avgErrSVM,np.std(errSVM),errSVM)
        dictMNB[FV_namesList]=(avgErrMNB,np.std(errMNB),errMNB)
        dictKNN[FV_namesList]=(avgErrSVM,np.std(errKNN),errKNN)
        dictDT[FV_namesList]=(avgErrDT,np.std(errDT),errDT)

        #keep lowest values
        lowRF=1
        lowDT=1
        lowKNN=1
        lowSVM=1
        lowMNB=1
        if avgErrRF < lowRF:
            lowRF = avgErrRF
            lowRF_fvs = FV_namesList
        #...
        if avgErrDT < lowDT:
            lowDT = avgErrDT
            lowDT_fvs = FV_namesList
        #...
        if avgErrKNN < lowKNN:
            lowKNN = avgErrKNN
            lowKNN_fvs = FV_namesList
        #...
        if avgErrSVM < lowSVM:
            lowSVM = avgErrSVM
            lowSVM_fvs = FV_namesList
        #...            
        if avgErrMNB < lowMNB:
            lowMNB = avgErrMNB
            lowMNB_fvs = FV_namesList
        #...            
    #... for FV_namesList in all_subsets(FV_names_combine):

    # save dictionaries
    if doSave:
        (FV_path,FV_fNameExt) = os.path.split(FV_file)
        os.chdir(FV_path)
        # save dictionaries to files
        with open('dictRF_totalNoW.csv', 'w') as f:
            [f.write('{0},{1}\n'.format(key, value)) for key, value in dictRF.items()]
        with open('dictKNN_totalNoW.csv', 'w') as f:
            [f.write('{0},{1}\n'.format(key, value)) for key, value in dictKNN.items()]
        with open('dictMNB_totalNoW.csv', 'w') as f:
            [f.write('{0},{1}\n'.format(key, value)) for key, value in dictMNB.items()]
        with open('dictSVM_totalNoW.csv', 'w') as f:
            [f.write('{0},{1}\n'.format(key, value)) for key, value in dictSVM.items()]
        with open('dictDT_totalNoW.csv', 'w') as f:
            [f.write('{0},{1}\n'.format(key, value)) for key, value in dictDT.items()]

        '''
        # save list of best features to file, with errors
        with open('bestRF2.csv', 'w') as f:
            [f.write(str(lowRF_fvs)+','+str(dictRF[lowRF_fvs]))]
        with open('bestKNN2.csv', 'w') as f:
            [f.write(str(lowKNN_fvs)+','+str(dictKNN[lowKNN_fvs]))]
        with open('bestMNB2.csv', 'w') as f:
            [f.write(str(lowMNB_fvs)+','+str(dictMNB[lowMNB_fvs]))]
        with open('bestSVM2.csv', 'w') as f:
            [f.write(str(lowSVM_fvs)+','+str(dictSVM[lowSVM_fvs]))]
        with open('bestDT2.csv', 'w') as f:
            [f.write(str(lowDT_fvs)+','+str(dictDT[lowDT_fvs]))]
        '''
    #...

    # print best feature sets for each classifier, with error rate
    print 'Random Forest best: ', lowRF,' features: ', lowRF_fvs
    print 'Decision tree best: ', lowDT,' features: ', lowDT_fvs
    print 'KNN best: ', lowKNN,' features: ', lowKNN_fvs
    print 'MNB best: ', lowMNB,' features: ', lowMNB_fvs
    print 'SVM best: ', lowSVM,' features: ', lowSVM_fvs
#... for baseFV in baseFVS:
