import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skimage
from skimage.transform import rotate
from skimage import morphology
from skimage import measure
import math

TRAIN = '../data/training/' 
VALID = '../data/validation/'
TEST = '../data/test/'

IMG = 'example_image/'
SEG = 'example_segmentation/'
FEAT = 'features/'
TRUTH = 'ground_truth.csv'

def get_boundaries(image):
    """Function to locate the boundaries of the lesion over the whole image.
    Takes a segmentation mask image as argument and returns the upper, lower, left and right boundaries."""

    mask = np.where(image == 1)
    left = min(mask[1])
    right = max(mask[1])
    upper = min(mask[0])
    lower = max(mask[0])
    return upper, lower, left, right

def get_center(image): # NOT NEEDED ANYMORE ?
    """Function that takes an image as input, and returns the centerpoint of the lesion."""
    up, dw, lt, rt = get_boundaries(image)
    center = ((up+dw)/2, (lt+rt)/2)
    return center
    
def zoom(image):
    """Function to zoom-in (crop) the lesion from blank space. Takes a segmentation mask image as input,
    and returns the rectangle where the lesion is found."""

    up, dw, lt, rt = get_boundaries(image)
    rectangle = image[up:dw+1, lt:rt+1]
    return rectangle

def cuts(image):
    """Function to perform a double cut (vertical and horizontal) of the lesion. Takes a segmentation mask image as input,
    and returns the vertical and horizontal cuts (2 for each dimension). It handles uneven shapes."""

    center_h = image.shape[0] // 2 # The image shape contains a tuple with height and width (in pixels)
    if image.shape[0] % 2 == 0: # If the height is an even number of pixels, the cut returns 2 equal sides
        upside = image[:center_h,:]
        downside = image[center_h:,:]
    else: # If the height is an uneven number of pixels, the cut has to "share" the center, to return 2 equal sides
        upside = image[:center_h,:]
        downside = image[center_h+1:,:]
        
    center_w = image.shape[1] // 2    
    if image.shape[1] % 2 == 0:
        leftside = image[:,:center_w]
        rightside = image[:,center_w:]
    else:
        leftside = image[:,:center_w]
        rightside = image[:,center_w+1:]
    return upside, downside, leftside, rightside


def test_symmetry(image, rot_deg=5):
    """Function to test the symmetry of an image. Takes a segmentation mask image and the rotation degree interval and
    returns a symmetry score between zero (non-symmetric) to one (completely symmetric)."""

    assert (rot_deg <= 90) and (rot_deg >= 0), "Rotation degree should be positive and at most 90 deg"
    optimal = 0

    for deg in range(0,91, rot_deg):
        rot_image = skimage.transform.rotate(image, deg, resize=True)
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
    """Function to convert a RGB image to grayscale."""
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def crop(image, mask, resize=True, warning=True):
    if image.shape[:2] != mask.shape[:2]:
        if warning:
            print("Image and Mask must have the same size. OPERATION CANCELLED.")
        else: return
    else:
        img = image.copy()
        img[mask==0] = 0

        if resize:
            u,d,l,r = get_boundaries(mask)
            img = img[u:d,l:r,...]
        return img

def color_std(image):
    """A function that takes an image as input, computes and returns the average standard deviation of all the
    rgb color values."""
    R = image[np.where(image[:,:,0] != 0) and np.where(image[:,:,1] != 0) and np.where(image[:,:,2] != 0)][:,0]
    G = image[np.where(image[:,:,0] != 0) and np.where(image[:,:,1] != 0) and np.where(image[:,:,2] != 0)][:,1]
    B = image[np.where(image[:,:,0] != 0) and np.where(image[:,:,1] != 0) and np.where(image[:,:,2] != 0)][:,2]
    color_std = (np.std(R) + np.std(G) + np.std(B)) /3
    #except:
    #    color_std = 'NA'
    return color_std

def check_border(image, border=0.01, tolerance=0.2, warning=True):
    """Function to check if the lesion might be exceeding the image. Take the following arguments:
    - image: segmentation mask image to check.
    - border: the percentage of pixels to consider as a border. 10% by default.
    - tolerance: the percentage of tolerance for a lesion to be at the border of the image. 20% by default.
    - warning: boolean to indicate if a textual warning should be issue when checking the border. True by default."""
    h = int(image.shape[0] * border)
    w = int(image.shape[1] * border)
    up = (np.sum(image[h,:]) / image.shape[1]) > tolerance
    dw = (np.sum(image[-h,:]) / image.shape[1]) > tolerance
    lt = (np.sum(image[:,w]) / image.shape[0]) > tolerance
    rt = (np.sum(image[:,w]) / image.shape[0]) > tolerance
    if warning:
        if up or dw or lt or rt: return "This lesion might be overflowing the image"
        else: return "This lesion does not seem to be overflowing the image"
    else:
        return up or dw or lt or rt


def masker(image, sens):
    '''Takes image, converts to a grayscale image, and returns a masked 
    image that only shows values below the sensitivity given as input.'''
    
    gray = rgb2gray(image) # Create grayscale image
    img2 = gray < sens # **This level needs manually adjusting, also need to be able to automate**
    
    # use plt.imshow(masker(image,sens), cmap='gray') to see image
    
    return img2.astype(int)

def dimensions(mask1):
    '''calculates height(max) and width(90 deg to height)
        returns height, width, rotated mask image, degree of rotation'''
    pixels_in_col = np.max(np.sum(mask1, axis=0))

    rot = 0
    max_col = 0
    rot_max = 0
    for _ in range(9):
        rot_im = transform.rotate(mask1,rot)
        pixels_in_col = np.max(np.sum(rot_im, axis=0))
        if pixels_in_col > max_col:
            max_col = pixels_in_col
            rot_max = rot
            pixels_in_row = np.max(np.sum(rot_im, axis=1))
        rot += 10

    return max_col, pixels_in_row, rot_max

def measure_area_perimeter(mask):
    """A function that takes either a segmented image or perimeter 
    image as input, and calculates the length of the perimeter of a lesion."""
    
    # Measure area: the sum of all white pixels in the mask image
    area = np.sum(mask)

    # Measure perimeter: first find which pixels belong to the perimeter.
    perimeter = measure.perimeter(mask)
    
    return area, perimeter

def predict(bi_image): # Predict might be a little confusing ?
    
    area = np.sum(bi_image)
    _, peri = perimeter(bi_image)
    
    area_from_peri = pi*((peri/(2*pi))**2)
    peri_from_area = 2*pi*sqrt(area/pi)
    
    return area, area_from_peri, peri, peri_from_area  

def main():
    test = pd.read_csv(TEST + TRUTH, index_col = "image_id")
    visual = pd.read_csv("../to_check.csv")
    ignore = visual[visual.loc[:,'Visual inspection'] == 'Ignore']

    for ix, row in ignore.iterrows():
        image = row.loc["image_id"]
        try:
            test.drop(image, axis=0, inplace=True)
        except: pass

    DoBatch = int(input("How many batches? "))

    batch = int(input("Do batch # "))

    length = test.shape[0] // DoBatch
    start = length * (batch - 1)
    end = length * (batch)

    outfile = open("symmetry.csv", "w")

    i = 1
    for ix, row in test[start:end].iterrows():
        file_path = TEST + SEG + str(ix) + "_segmentation.png"
        image = plt.imread(file_path)

        ptg = round(i / length,2)
        print(f'\rCalculating symmetry: {ptg:.2%}', end='\r')

        outfile.write(ix + "," + str(test_symmetry(image)) + "\n")
        
        i += 1

    outfile.close()

if __name__ == "__main__":
    main()