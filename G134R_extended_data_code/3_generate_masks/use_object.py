import apoc
import os
from skimage.io import imread, imsave
import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt

channel = "Cell"

image_folder = r"Z:/personal_data/Filip_Liebsch/SP8_Biocenter_small/2025-04-23_floxed/S325D_project/training/" + channel + "/images/"
file_list = os.listdir(image_folder)

# this is the name of the classifier saved
cl_filename = channel + '_object_model_optm.cl'

# re-load segmenter
segmenter = apoc.ObjectSegmenter(opencl_filename=cl_filename)
# show all images
for i, filename in enumerate(file_list):
    fig, axs = plt.subplots(1, 2, figsize=(15, 15))

    image = imread(image_folder + filename)
    cle.imshow(image, plot=axs[0])

    labels = segmenter.predict(image)
    cle.imshow(labels, plot=axs[1], labels=True)

    plt.show()

print("...completed.")
