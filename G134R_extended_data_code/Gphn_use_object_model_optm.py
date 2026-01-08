import apoc
import os
from skimage.io import imread, imsave
import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt

channel = "Gphn"

image_folder = r"Z:/personal_data/Filip_Liebsch/SP8_Biocenter_small/2025-04-23_floxed/S325D_project/data/2025-03-31/S325D/cell1/Z-projection"
image_name = "MaxZ_deconvolution.tif"
image_path = os.path.join(image_folder, image_name)
image = imread(image_path)

channel = image [:,:,2]  # assuming channel 3 is Gphn

# this is the name of the classifier saved
cl_filename = "Gphn_object_model_optm.cl"

# re-load segmenter
segmenter = apoc.ObjectSegmenter(opencl_filename=cl_filename)


out_path = os.path.join(image_folder, image_name.replace('MaxZ_deconvolution.tif', 'Gphn_labels.tif'))

labels = segmenter.predict(channel)

imsave(out_path, labels)

print("...completed.")
