# Import the required modules
# Import the required modules
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import argparse as ap
import glob
import os
from config import *
import numpy as np
import pickle

if __name__ == "__main__":
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--posfeat", help="Path to the positive features directory", required=True)
    parser.add_argument('-n', "--negfeat", help="Path to the negative features directory", required=True)
    parser.add_argument('-c', "--classifier", help="Classifier to be used", default="LIN_SVM")
    args = vars(parser.parse_args())

    pos_feat_path =  args["posfeat"]
    neg_feat_path = args["negfeat"]

    # Classifiers supported
    clf_type = args['classifier']

    fds = []
    labels = []
    n_pos_train = 5000
    pos_feat = pickle.load(open(os.path.join(pos_feat_ph, 'pos.p'), 'rb'))
    # Load the positive features
    for i in list(range(n_pos_train,4000)):
        fd = pos_feat[i]
        fds.append(fd)
        if i == 9:
            print("Features size: " + str(fd.size))
        labels.append(1)
    del pos_feat

    neg_feat = pickle.load(open(os.path.join(neg_feat_ph, 'neg.p'), 'rb'))
    # Load the negative features
    n_neg_train = 5000
    for i in list(range(n_neg_train,20000)):
        fd = neg_feat[i]
        fds.append(fd)
        labels.append(0)
    del neg_feat

    print(type(fds))
    print(fds[0])
    fds_array = np.array(fds)
    labels_array = np.array(labels)
    print labels_array[labels_array>0]
    print(fds_array.shape)

    if clf_type is "LIN_SVM":
        clf = joblib.load(model_path)
        print "Training a Linear SVM Classifier"
        # preds = clf.predict(fds_array)
        # fpr, tpr, thresholds = metrics.roc_curve(labels_array, preds, pos_label=2)
        sc = clf.score(fds_array, labels_array)
        print("The score is: " + str(sc))
        # print(metrics.auc(fpr,tpr))
        # If feature directories don't exist, create them
