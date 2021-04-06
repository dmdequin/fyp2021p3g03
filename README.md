# fyp2021p3g03
# First Year Project - ITU 2021 - Project #3 - Group 3

## Project goals
In this project we will learn to measure features in images of skin lesions, and predict the diagnosis (for example, melanoma) from these features in an automatic way. We will likely:
* Implement methods to measure "handcrafted" features
* Predict the lesion diagnosis using simple classifiers, based on the features
* Perform experiments to test different parts of your method

## Project code
In this project we will work with Python, and jupyter notebook.

To start with, we have two files:
* fyp2021p3_group00_functions.py with functions to extract features etc.
* fyp2021p3_group00.py with the main script, which loads the images, calls the functions, and reproduces your results.

## Data
We can use two types of data for the project:

* Default images, available in this repository. These images are from the ISIC 2017 challenge.
* Images from other public repositories.

In general, Github is not very suitable for large collections of image files, so we may need to use other solutions (such as downloading a zip archive from an external source, and unzipping it before the script runs.

## ISIC 2017 data
The ISIC 2017 dataset is available via https://challenge.isic-archive.com/landing/2017. There are more than 2000 images available in this dataset. In this repository, only 150 images from this dataset are added, as a demonstration. The following is available per image:
* ISIC_[ID].png the image of the lesion
* ISIC_[ID]\_segmentation.png the mask of the lesion, showing which pixels belong to the lesion or not
The label of the image, i.e. whether it belongs to the Melanoma class (0 = no, 1 = yes), and/or the Keratosis class (0 = no, 1 = yes).

## Other data
(Mention other data we use)

## Final project
At the end we will produce a report and the code used.
