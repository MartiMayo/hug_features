import numpy as np
import os
import sys
import glob
import random
from skimage.feature import hog
from config import *
from scipy.misc import toimage

def obtain_hogs(im):
	im = toimage(im)
	im = np.asarray(im)
	fd = hog(im,  orientations, pixels_per_cell, cells_per_block, visualize, normalize)
	return fd

def load_patient(path):
	"""
	Loads the patient to a 4D-numpy array
	"""
	patient = np.load(path)['arr_0']
	return patient

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

def get_label(image, x, y, window_size, nodules_image, nodules_image_sum, total_area):
	"""
	Check whether the window fully contains at least half of the tumor (we assume a single tumor for slice)
	"""
	nodules_in_window = nodules_image[y: y + window_size[1], x: x + window_size[0]].sum()
	return (nodules_in_window >= total_area/2) | (nodules_in_window >= nodules_image_sum/2)

def window_area(window_size):
	return window_size[0] * window_size[1]

def is_inside_lung(x, y, window_size, lung_image):
	"""
	Check whether at least half of the window is inside the lung
	"""
	return lung_image[y: y + window_size[1], x: x + window_size[0]].sum() >= window_area(window_size) / 2

def is_inside_image(image, x, y, window_size):
	return ((y + window_size[1] <= image.shape[0]) and (x + window_size[0] <= image.shape[1]))

def patch_image_and_label(image, step_size, window_size, lung_image, nodules_image, nodules_image_sum, total_area):
	for y in xrange(0, image.shape[0], step_size[1]):
		for x in xrange(0, image.shape[1], step_size[0]):
			if (is_inside_image(image, x, y, window_size) and is_inside_lung(x, y, window_size, lung_image)):
				yield (x, y, image[y: y + window_size[1], x: x + window_size[0]], get_label(image, x, y, window_size, nodules_image, nodules_image_sum, total_area))

if __name__ == '__main__':
	total_area = window_area(min_wdw_sz)
	patient_id = 1
	for file_path in glob.glob(train_images_path + "*npz"):
		patient = load_patient(file_path)
		if (patient.shape[0] < 3):
			print("Patient " + file_path.split('/')[-1] + " does not have the nodule level so we skip him")
			continue
		slices_with_nodules = obtain_slices_with_nodules(patient)
		if (slices_with_nodules):
			fds_out = []
			print("Patient " + file_path.split('/')[-1] + " has these slices with nodules")
			print(slices_with_nodules)
			print("Patching the images...")
			for si, sl in enumerate(slices_with_nodules):
				level0_image = patient[0, sl]
				level1_image = patient[1, sl]
				level2_image = patient[2, sl]
				patched_image = patch_image_and_label(level0_image, step_size, min_wdw_sz, level1_image, level2_image, level2_image.sum(), total_area)
				list_patches = list(patched_image)
				print("Dumping the results: slice " + str(si+1) + " out of " + str(len(slices_with_nodules)))
				for i, patch in enumerate(list_patches):
					if i % 2000 == 0:
						print("Patch " + str(i) + " out of " + str(len(list_patches)))
					label = np.array([float(patch[3] == True)])
					im = patch[2]
					fd = obtain_hogs(im)
					fd = np.concatenate((label,fd))
					fds_out.append(fd)
			fds_out = np.stack(fds_out)
			print fds_out.shape
			np.savez(hog_features_path + str(patient_id) + '.npz', fds_out)
			patient_id += 1
					#np.savez(pos_im_patch + str(pos_counter) + '.npz', patch[2])
		else:
			print("Patient " + file_path + " has no slices with nodules")
