#!/usr/bin/python
import os
from config import *

# Get the images of the initial dataset and turn them into labeled patches
#os.system("python ../object-detector/sliding_window.py")

# Perform training
os.system("python train-classifier.py")

# Perform testing
#test_im_path = "/home/mmc/code/hog_features/data/images/testImage.npy"
#os.system("python ../object-detector/test-classifier.py -i {} --visualize".format(test_im_path))

# Score dataset
#os.system("python score.py")
