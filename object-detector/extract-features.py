# Import the functions to calculate feature descriptors
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
# To read file names
import argparse as ap
import glob
import os
import numpy as np
from config import *
from scipy.misc import toimage
import pickle

if __name__ == "__main__":
    # Argument Parser
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--pospath", help="Path to positive images",
            required=True)
    parser.add_argument('-n', "--negpath", help="Path to negative images",
            required=True)
    parser.add_argument('-d', "--descriptor", help="Descriptor to be used -- HOG",
            default="HOG")
    args = vars(parser.parse_args())

    pos_im_path = args["pospath"]
    neg_im_path = args["negpath"]

    des_type = args["descriptor"]

    # If feature directories don't exist, create them
    if not os.path.isdir(pos_feat_ph):
        os.makedirs(pos_feat_ph)

    # If feature directories don't exist, create them
    if not os.path.isdir(neg_feat_ph):
        os.makedirs(neg_feat_ph)

    list_pos = []
    print "Calculating the descriptors for the positive samples and saving them"
    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        im = np.load(im_path)
        key = im.keys()[0]
        im = im[key]
        im = toimage(im)
        im = np.asarray(im)
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            list_pos.append(fd)
    pickle.dump(list_pos, open(os.path.join(pos_feat_ph, 'pos.p'), 'wb'))
    del list_pos
    print "Positive features saved in {}".format(pos_feat_ph)

    list_neg = []
    print "Calculating the descriptors for the negative samples and saving them"
    for im_path in glob.glob(os.path.join(neg_im_path, "*")):
        im = np.load(im_path)
        key = im.keys()[0]
        im = im[key]
        im = toimage(im)
        im = np.asarray(im)
        if des_type == "HOG":
            fd = hog(im,  orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            list_neg.append(fd)
    pickle.dump(list_neg, open(os.path.join(neg_feat_ph, 'neg.p'), 'wb'))
    print "Negative features saved in {}".format(neg_feat_ph)

    print "Completed calculating features from training images"
