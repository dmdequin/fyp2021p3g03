"""
Functions for the FYP 2021 project 3
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import morphology
from scipy.spatial.distance import cdist
from scipy.stats.stats import mode


def scatter_data(x1, x2, y, ax=None):
    # scatter_data displays a scatterplot of featuress x1 and x2, and gives each point
    # a different color based on its label in y

    class_labels, indices1, indices2 = np.unique(y, return_index=True, return_inverse=True)
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.grid()

    colors = cm.rainbow(np.linspace(0, 1, len(class_labels)))
    for i, c in zip(np.arange(len(class_labels)), colors):
        idx2 = indices2 == class_labels[i]
        lbl = 'Class ' + str(i)
        ax.scatter(x1[idx2], x2[idx2], color=c, label=lbl)

    return ax


def measure_area_perimeter(mask):
    # Measure area: the sum of all white pixels in the mask image
    area = np.sum(mask)

    # Measure perimeter: first find which pixels belong to the perimeter.
    struct_el = morphology.disk(1)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    image_perimeter = mask - mask_eroded

    # Now we have the perimeter image, the sum of all white pixels in it
    perimeter = np.sum(image_perimeter)

    return area, perimeter


def knn_classifier(x_train, y_train, x_validation, x_test, k):
    # Returns the labels for test_data, predicted by the k-NN clasifier trained on X_train and y_train
    # Input:
    # X_train - num_train x num_features matrix with features for the training data
    # y_train - num_train x 1 vector with labels for the training data
    # X_validation - num_test x num_features matrix with features for the validation data
    # X_test - num_test x num_features matrix with features for the test data
    # k - Number of neighbors to take into account
    # Output:
    # y_pred_validation - num_test x 1 predicted vector with labels for the validation data
    # y_pred_test - num_test x 1 predicted vector with labels for the test data

    x_test_val = np.vstack((x_validation, x_test))
    
    # Compute standardized euclidian distance of validation and test points to the other points
    distances = cdist(x_test_val, x_train, metric='seuclidean')
    
    # Sort distances per row and return array of indices from low to high
    sort_ix = np.argsort(distances, axis=1)
    
    # Get the k smallest distances
    sort_ix_k = sort_ix[:, :k]
    predicted_labels = y_train[sort_ix_k]
    
    # Predictions for each point is the mode of the K labels closest to the point
    predicted_labels = mode(predicted_labels, axis=1)[0]
    y_pred_validation = predicted_labels[:len(x_validation)]
    y_pred_test = predicted_labels[len(x_validation):]
    
    return y_pred_validation, y_pred_test
"""
def get_boundaries(image):
	"""
	#Takes a segmentation mask image as input and finds the boundaries of the mask;
	#meaning the extremes of the segmented mask.
	#It returns the upper, lower, left and right boundaries.
	"""
    mask = np.where(image == 1)
    left = min(mask[1])
    right = max(mask[1])
    upper = min(mask[0])
    lower = max(mask[0])
    return upper, lower, left, right
"""
def get_center(image): # NOT NEEDED ANYMORE ?
	"""
	Takes a segmentation mask image as input and finds the center of the mask.
	"""
    up, dw, lt, rt = get_boundaries(image)
    center = ((up+dw)/2, (lt+rt)/2)
    return center
    
def zoom(image):
	"""
	Takes a segmentation mask image as input and returns the zoomed-in rectangle where the mask is present.
	"""
    up, dw, lt, rt = get_boundaries(image)
    rectangle = image[up:dw+1, lt:rt+1]
    return rectangle

def cuts(image):
	"""
	Takes a sementation mask image as input and returns four cuts:
		- Upside: The upper side of a horizontal cut from the center of the mask
		- Downside: The lower side of a horizontal cut from the center of the mask
		- Leftside: The left side of a vertical cut from the center of the mask
		- Rightside: The right side of a vertical cut from the center of the mask
	"""
    center_h = image.shape[0] // 2 # The image shape contains a tuple with height and width (in pixels)
    if image.shape[0] % 2 == 0: # If the height is an even number of pixels, the cut returns 2 equal sides
        upside = image[:center_h,:]
        downside = image[center_h:,:]
    else: # If the height is an uneven number of pixels, the cut has to "share" the center, to return 2 equal sides
        upside = image[:center_h+1,:]
        downside = image[center_h:,:]
        
    center_w = image.shape[1] // 2    
    if image.shape[1] % 2 == 0:
        leftside = image[:,:center_w]
        rightside = image[:,center_w:]
    else:
        leftside = image[:,:center_w+1]
        rightside = image[:,center_w:]
 
    return upside, downside, leftside, rightside


def test_symmetry(image, rot_deg=30):
    """
	This function takes a segmented mask image and rotation degree as arguments,
	and returns a symmetry score between 0 (not symmetric) and 1 (totally symmetric).
	The function rotates the image to find the best possible symmetry.
    """
    assert (rot_deg <= 90) and (rot_deg >= 0), "Rotation degree should be positive and at most 90 deg"
    optimal = 0
    
    for deg in range(0,90, rot_deg):
        rot_image = skimage.transform.rotate(image, deg)
        z = zoom(rot_image)
        
        upside, downside, leftside, rightside = cuts(z)

        up_dw = np.sum(np.bitwise_and(upside.astype(int), np.flipud(downside).astype(int))) /\
        np.sum(np.bitwise_or(upside.astype(int), np.flipud(downside).astype(int)))

        lt_rt = np.sum(np.bitwise_and(leftside.astype(int), np.fliplr(rightside).astype(int))) /\
        np.sum(np.bitwise_or(leftside.astype(int), np.fliplr(rightside).astype(int)))
    
        symmetry = (up_dw+lt_rt)/2
        
        if symmetry > optimal: optimal = symmetry

    return symmetry
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_mask(image):
    gray = rgb2gray(image)
    plt.hist(gray)

#def crop(image, mask):
#    img = image.copy()
#    img[mask==0] = 0
#    return img

def color_std(image):
    try:
        R = image[np.where(image[:,:,0] != 0) and np.where(image[:,:,1] != 0) and np.where(image[:,:,2] != 0)][:,0]
        G = image[np.where(image[:,:,0] != 0) and np.where(image[:,:,1] != 0) and np.where(image[:,:,2] != 0)][:,1]
        B = image[np.where(image[:,:,0] != 0) and np.where(image[:,:,1] != 0) and np.where(image[:,:,2] != 0)][:,2]
        color_std = (np.std(R) + np.std(G) + np.std(B)) /3
    except:
        color_std = 'NA'
    return color_std