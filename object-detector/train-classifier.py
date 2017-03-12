# Import the required modules
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import argparse as ap
import glob
import os
from config import *
import numpy as np
import pickle
from sklearn import metrics
import random
def load_patient(path):
    """
    Loads the patient to a 4D-numpy array
    """
    patient = np.load(path)['arr_0']
    return patient

def sample_patient(patient_data):
    index_pos = (patient_data[:,0] == 1)
    n_pos = sum(index_pos)
    pos_patients = patient_data[index_pos,:]
    neg_patients = patient_data[~index_pos,:]
    idx = np.random.choice(len(neg_patients),size = n_pos,replace = True)
    neg_patients = neg_patients[idx,:]
    tot_patients = np.concatenate((pos_patients,neg_patients),axis = 0)
    return tot_patients
def split_train(dataset):
    labels = dataset[:,0]
    dataset = np.delete(dataset,0,axis = 1)
    return dataset, labels

if __name__ == "__main__":
    # Classifier type

    random.seed(20)
    # Loading patients
    training_set = []
    i = 0
    for file_path in glob.glob(hog_features_path + "*npz"):
        patient_data = load_patient(file_path)
        patient_data = sample_patient(patient_data)
        patient_id = np.zeros((len(patient_data),1)) + i
        patient_data = np.concatenate((patient_id,patient_data),axis = 1)
        training_set.append(patient_data)
        i +=1

    fds_total = np.vstack(training_set)
    np.savetxt("features.csv",fds_total,delimiter = ";")
    print fds_total.shape
    index_t=fds_total[:,0]<17
    print np.unique(fds_total[:,0])
    fds_total = np.delete(fds_total,0,axis=1)
    fds_train,labels_train = split_train(fds_total[index_t,:])
    fds_test,labels_test = split_train(fds_total[~index_t,:])
    clf = RandomForestClassifier(n_estimators=200, n_jobs = 3)
    clf = GradientBoostingClassifier(n_estimators = 200)
    #clf = LogisticRegression()
    #clf = LinearSVC()
    print "Training ..."
    clf.fit(fds_train, labels_train)
    # If feature directories don't exist, create them

    preds = clf.predict_proba(fds_train)[:,1]
    #preds = clf.predict(fds_train)
    threshold = 0.55
    fpr, tpr, thresholds = metrics.roc_curve(labels_train, preds, pos_label=1)
    sc = clf.score(fds_train, labels_train)
    print("The score is: " + str(sc))
    print("The auc is: " + str(metrics.auc(fpr,tpr)))
    #print sum(labels_train[preds > threshold]/sum(labels_train))

    # Hard negative mining    
    index_neg = (labels_train == 0) & (preds > threshold)
    #index_pos = (labels_train == 1) & (preds < threshold)
    index_total = (index_neg) #| (index_pos)
    fds_total = np.concatenate((fds_train,fds_train[index_total]), axis = 0)
    labels_total = np.concatenate((labels_train,labels_train[index_total]), axis = 0)
    clf.fit(fds_total,labels_total)
    if not os.path.isdir(os.path.split(model_path)[0]):
        os.makedirs(os.path.split(model_path)[0])
    joblib.dump(clf, model_path)

    print "Classifier saved to {}".format(model_path)
    preds = clf.predict_proba(fds_train)[:,1]
    #preds = clf.predict(fds_train)
    fpr, tpr, thresholds = metrics.roc_curve(labels_train, preds, pos_label=1)
    sc = clf.score(fds_train, labels_train)
    print("The score is: " + str(sc))
    print("The auc is: " + str(metrics.auc(fpr,tpr)))
    #print sum(labels_train[preds > threshold]/sum(labels_train))

    # Validation

    print "Validation"
    #preds = clf.predict_proba(fds_test)[:,1]
    preds = clf.predict(fds_test)
    fpr, tpr, thresholds = metrics.roc_curve(labels_test, preds, pos_label=1)
    sc = clf.score(fds_test, labels_test)
    print("The score is: " + str(sc))
    print("The auc is: " + str(metrics.auc(fpr,tpr)))
    print sum(labels_test[preds > threshold]/sum(labels_test))