# Import the required modules
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import argparse as ap
import glob
import os
from config import *
import numpy as np
import pickle
from sklearn import metrics
import random

if __name__ == "__main__":
    # Classifier type
    clf_type = "LIN_SVM"

    random.seed(20)

    pos_feat = pickle.load(open(os.path.join(pos_feat_path, 'pos.p'), 'rb'))
    pos_feat = np.vstack(pos_feat)
    n_pos_train = 15000
    idx_pos =  np.random.choice(len(pos_feat), size=n_pos_train,replace = False)
    idx_pos = pos_feat[:,0]<20
    pos_feat = np.delete(pos_feat, 0, axis = 1)
    print pos_feat.shape

    neg_feat = pickle.load(open(os.path.join(neg_feat_path, 'neg.p'), 'rb'))
    neg_feat = np.vstack(neg_feat)
    #neg_feat = neg_feat[np.arange(15000),:]
    n_neg_train = 15000
    idx_neg = np.random.choice(len(neg_feat),size = n_neg_train,replace = True) + len(pos_feat)
    idx_neg = neg_feat[:,0] < 20
    neg_feat = np.delete(neg_feat,0,axis = 1)

    idx = np.concatenate((idx_pos,idx_neg),axis = 0)
    fds = np.concatenate((pos_feat,neg_feat),axis = 0)
    labels = np.concatenate((np.full((len(pos_feat)),1),np.full((len(neg_feat)),0)),axis = 0)


    # Training
    fds_train = fds[idx,:]
    labels_train = labels[idx]
    # Testing
    fds_test = fds[~idx,:]
    labels_test = labels[~idx]
    clf = RandomForestClassifier(n_estimators=200, n_jobs = 5)
    print "Training ..."
    clf.fit(fds_train, labels_train)
    # If feature directories don't exist, create them

    preds = clf.predict_proba(fds_test)[:,1]
    threshold = 0.55
    fpr, tpr, thresholds = metrics.roc_curve(labels_test, preds, pos_label=1)
    sc = clf.score(fds_test, labels_test)
    print("The score is: " + str(sc))
    print("The auc is: " + str(metrics.auc(fpr,tpr)))
    print sum(labels_test[preds > threshold]/sum(labels_test))

    # Hard negative mining    
    index_neg = (labels_test == 0) & (preds > threshold)
    index_pos = (labels_test == 1) & (preds < threshold)
    index_total = (index_neg) | (index_pos)
    fds_total = np.concatenate((fds_train,fds_test[index_total]), axis = 0)
    labels_total = np.concatenate((labels_train,labels_test[index_total]), axis = 0)
    clf.fit(fds_total,labels_total)
    if not os.path.isdir(os.path.split(model_path)[0]):
        os.makedirs(os.path.split(model_path)[0])
    joblib.dump(clf, model_path)

    print "Classifier saved to {}".format(model_path)
    preds = clf.predict_proba(fds_test)[:,1]

    fpr, tpr, thresholds = metrics.roc_curve(labels_test, preds, pos_label=1)
    sc = clf.score(fds_test, labels_test)
    print("The score is: " + str(sc))
    print("The auc is: " + str(metrics.auc(fpr,tpr)))
    print (labels_test[preds > threshold])
    print sum(labels_test[preds > threshold]/sum(labels_test))
