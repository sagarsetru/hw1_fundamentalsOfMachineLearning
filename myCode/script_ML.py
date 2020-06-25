import numpy as np
import random as random
import scipy.stats as stats
import scipy.io as spio
import os
import itertools

#import classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

# import miscellaneous tools
from sklearn.metrics import zero_one_loss
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix

# load labels
LBs = '/Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/myData/fisherVectors/numClusters10_exemplarSize3/LB.mat'
labels = spio.loadmat(LBs)['LB'][0]

# choose fisher vectors to load
FV_names = ('brightness', 'mfc', 'chroma', 'eng', 'keystrength', 'roughness', 'zerocross', 'combined')
FV_names = ('zerocross', 'combined') # note: need to include at least one other feature if doing combined only
FV_names_combine = ('brightness', 'mfc', 'chroma', 'eng', 'keystrength', 'roughness', 'zerocross')
FV_names = ('mfc,')
# function to make all combinations
def all_subsets(ss):
  return itertools.chain(*map(lambda x: itertools.combinations(ss, x), range(0, len(ss)+1)))


# choose classifier to run
doSVM = 0
doKNN = 0
doRF = 0

## SAVE DATA
doSave = 0

for FV_name in FV_names:
    print 'Current FV: ', FV_name

    if FV_name is not 'combined':
        #FVs = '/Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/myData/fisherVectors/numClusters10_exemplarSize3/mfc.mat'
        FV_file = '/Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/myData/fisherVectors/numClusters10_exemplarSize5/'+FV_name+'.mat'

        # load into arrays
        FVs = spio.loadmat(FV_file)['FV']
        N = FVs.shape[1]
    else:
        counter = -1
        for FV_name_combine in FV_names_combine:
            counter += 1
            FV_file = '/Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/myData/fisherVectors/numClusters10_exemplarSize3/'+FV_name_combine+'.mat'
            if counter is 0:
                FVs = spio.loadmat(FV_file)['FV']
            else:
                FVs_current = spio.loadmat(FV_file)['FV']
                FVs = np.concatenate((FVs,FVs_current))
            #...
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
    CM_RF = np.zeros((10,10))

    # KNN learning
    k = 10
    avgErrKNN = 0.0
    errKNN = np.zeros(k_folds)
    CM_KNN = np.zeros((10,10))

    # multinomial NB learning
    #avgErrMNB = 0.0
    #errMNB = np.zeros(k_folds)
    #CM_MNB = np.zeros((10,10))

    # svm
    avgErrSVM = 0.0
    errSVM = np.zeros(k_folds)
    CM_SVM = np.zeros((10,10))

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
            rf = RandomForestClassifier(n_estimators = maxLearners, max_depth = maxDepth, warm_start = False)
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

        # multinomial naive bayes
        #mnb = MultinomialNB()
        #mnb.fit(X_TR.T,Y_TR)
        #predMNB = mnb.predict(X_TE.T)
        #errMNB[counter] = zero_one_loss(predMNB,Y_TE)
        #print 'Error from multinomial Naive Bayes: ', errMND[counter]
        #avgErrMNB += (1./k_folds) * errMNB[counter]
        # determine confusion matrix   
        #CM_MNB += confusion_matrix(predRF,Y_TE)

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
        print 'Average error from Random Forest Classifier:'
        print '%4.2f%s' % (100 * avgErrRF, '%')
    #...
    if doKNN:
        print 'Average error from KNN:'
        print '%4.2f%s' % (100 * avgErrKNN, '%')
    #...
    if doSVM:
        print 'Average error from SVM:'
        print '%4.2f%s' % (100 * avgErrSVM, '%')
    #...
    print ' '

    if doSave:
        if FV_name is not 'combined':
            # generate strings for directory to save data
            (FV_path,FV_fNameExt) = os.path.split(FV_file)
            (FV_fName,FV_ext) = os.path.splitext(FV_fNameExt)
            saveDir = FV_path+'/'+FV_fName+'_learning'
            if not os.path.isdir(saveDir):
                os.makedirs(saveDir)
            #...
            os.chdir(saveDir)
        else:
            # generate strings for directory to save data
            (FV_path,FV_fNameExt) = os.path.split(FV_file)
            #(FV_fName,FV_ext) = os.path.splitext(FV_fNameExt)
            saveDir = FV_path+'/'+'combined'+'_learning'
            if not os.path.isdir(saveDir):
                os.makedirs(saveDir)
            #...
            os.chdir(saveDir)
        #...
        
        # save error files and confusion matrices for each classifier
        spio.savemat('rf.mat',{'errRF':errRF,'CM_RF':CM_RF})
        spio.savemat('knn.mat',{'errKNN':errKNN,'CM_KNN':CM_KNN})
        spio.savemat('svm.mat',{'errSVM':errSVM,'CM_SVM':CM_SVM})
    #...
#...
