import numpy as np
import os
import sys
import glob
import random
from scipy.misc import toimage
from PIL import Image
def obtain_slices_with_nodules(patient):
	"""
	Given a patient obtains the slides with nodules in them
	"""
	slices = []
	for i in range(patient.shape[1]):
		nodules_mask = patient[2,i]
		if nodules_mask.sum()!=0:
			slices.append(i)
	return slices

def load_patient(path):
	"""
	Loads the patient to a 4D-numpy array
	"""
	patient = np.load(path)['arr_0']
	return patient

path_to_images = 'D:/hug_features/luna_preprocess/'
path_to_positives = 'D:/hug_features/data/luna_preprocess/positives/'
path_to_negatives = 'D:/hug_features/data/luna_preprocess/negatives/'
file_path = glob.glob(path_to_images + "*npz")[21]
patient = load_patient(file_path)
slices_with_nodules = obtain_slices_with_nodules(patient)
slice_nd = slices_with_nodules[0]
toimage(patient[0,slice_nd]).save("D:/hug_features/test.bmp")
toimage(patient[0,slice_nd]).show()
toimage(patient[2,slice_nd]).show()
np.savez("D:/hug_features/test.npz",patient[0,slice_nd])