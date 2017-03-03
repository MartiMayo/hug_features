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
    for i in list(range(0,n_pos_train)):
        fd = pos_feat[i]
        fds.append(fd)
        labels.append(1)
    del pos_feat

    neg_feat = pickle.load(open(os.path.join(neg_feat_ph, 'neg.p'), 'rb'))
    # Load the negative features
    n_neg_train = 5000
    for i in list(range(0,n_neg_train)):
        fd = neg_feat[i]
        fds.append(fd)
        labels.append(0)
    del neg_feat

    print(type(fds))
    print(fds[0])
    fds_array = np.array(fds)
    labels_array = np.array(labels)
    print(fds_array.shape)

    if clf_type is "LIN_SVM":
        clf = LinearSVC()
        print "Training a Linear SVM Classifier"
        clf.fit(fds_array, labels_array)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        joblib.dump(clf, model_path)
        print "Classifier saved to {}".format(model_path)
