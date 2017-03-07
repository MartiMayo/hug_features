import numpy as np
import argparse as ap
import glob
from sklearn.externals import joblib
from skimage.feature import hog
from scipy.misc import toimage
from config import *
from sliding_window import load_patient, is_inside_image, is_inside_lung
from nms import nms

def patch_image(image, step_size, window_size, lung_image):
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            if (is_inside_image(image, x, y, window_size) and is_inside_lung(x, y, window_size, lung_image)):
                yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

def process(clf, patient, step_size, window_size):
    """
    Returns a list of tuples (slide_id, list[detections])
    """
    patient_detections = []
    all_slices = patient[0]
    for slice_id, image in enumerate(all_slices):
        lung_image = patient[1,slice_id]
        patched_image = patch_image(image, step_size, window_size, lung_image)
        detections = []
        for (x, y, patch) in patched_image:
            patch = toimage(patch)
            patch = np.asarray(patch)
            fd = hog(patch, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            fd = fd.reshape(1,-1)
            pred = classifier.predict(fd)
            if pred == 1:
                print  "Detection:: Location -> ({}, {})".format(x, y)
                detections.append([x, y, clf.decision_function(fd), window_size[0], window_size[1]])
        if detections:
            detections = nms(detections)
            patient_detections.append((slice_id, detections))
    return patient_detections


if __name__ == '__main__':
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-d', "--path_dataset", help="Path to the dataset to be scored", required=True)
    args = vars(parser.parse_args())

    # Load the classifier
    classifier = joblib.load(model_path)

    path_dataset =  args["path_dataset"]
    print("The dataset with the images to score is:")
    print(path_dataset)
    for file_path in glob.glob(path_dataset + "*.npz"):
        patient = load_patient(file_path)
        print("Loaded patient " + file_path)
        detections = process(classifier, patient, step_size, min_wdw_sz)
        for slice_id, dt in detections:
            print("Detections in the slide " + str(slice_id) + ": ")
            for d in dt:
                print(str(d))
