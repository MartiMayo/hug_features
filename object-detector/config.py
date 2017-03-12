'''
Set the config variable.
'''

import ConfigParser as cp
import json

config = cp.RawConfigParser()
config.read('../data/config/config.cfg')

min_wdw_sz = json.loads(config.get("hog","min_wdw_sz"))
step_size = json.loads(config.get("hog", "step_size"))
orientations = config.getint("hog", "orientations")
pixels_per_cell = json.loads(config.get("hog", "pixels_per_cell"))
cells_per_block = json.loads(config.get("hog", "cells_per_block"))
visualize = config.getboolean("hog", "visualize")
normalize = config.getboolean("hog", "normalize")
hog_features_path = config.get("paths","hog_features_path")
pos_im_patch = config.get("paths", "pos_im_patch")
neg_im_patch = config.get("paths", "neg_im_patch")
pos_feat_path = config.get("paths", "pos_feat_path")
neg_feat_path = config.get("paths", "neg_feat_path")
model_path = config.get("paths", "model_path")
train_images_path = config.get("paths", "train_images_path")
score_images_path = config.get("paths", "score_images_path")
threshold = config.getfloat("nms", "threshold")
