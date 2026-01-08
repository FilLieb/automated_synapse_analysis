import apoc
import os
from skimage.io import imread
import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt

channel = "Cell"

image_folder = r"Z:/personal_data/Filip_Liebsch/SP8_Biocenter_small/2025-04-23_floxed/S325D_project/training/" + channel + "/images/"
masks_folder = r"Z:/personal_data/Filip_Liebsch/SP8_Biocenter_small/2025-04-23_floxed/S325D_project/training/" + channel + "/masks/"

#image_folder = r"C:/Users/Filip/Sciebo/training/" + channel + "/images/"
#masks_folder = r"C:/Users/Filip/Sciebo/training/" + channel + "/masks/"

# this is where the model will be saved
cl_filename = channel + '_object_model_optm.cl'
apoc.erase_classifier(cl_filename) # delete it if it was existing before

# setup classifier and where it should be saved
segmenter = apoc.ObjectSegmenter(opencl_filename=cl_filename,
                                 max_depth=5,
                                 num_ensembles=1000)

# setup feature set used for training
# Gpgn features = "difference_of_gaussian=5 laplace_box_of_gaussian_blur=5"
# Cell
features = "gaussian_blur=6 gaussian_blur=26"

# train classifier on folders
apoc.erase_classifier(cl_filename)
apoc.train_classifier_from_image_folders(
    segmenter,
    features,
    image = image_folder,
    ground_truth = masks_folder)

print("Training completed and model saved to " + cl_filename)
