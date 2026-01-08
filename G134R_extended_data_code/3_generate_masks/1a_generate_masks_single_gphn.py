import apoc
import os
from skimage.io import imread, imsave

import numpy as np
import pandas as pd 
import pyclesperanto_prototype as cle

from skimage import measure
from skimage.segmentation import clear_border
from skimage.filters import threshold_li
from skimage.restoration import rolling_ball 

from scipy.ndimage import median_filter
import napari_simpleitk_image_processing as nsitk

# Set paths
image_folder = "G134R_extended_data_code/example_data/single_gphn/"
image_name = "MaxZ_deconvolution.tif"
confocal_name = "AvZ_confocal.tif"

def execute(root):
    print(f"Found image in: {root}")
    image_path = os.path.join(root, image_name)
    image = imread(image_path)

    confocal_path = os.path.join(root, confocal_name)
    confocal = imread(confocal_path)

    folder_name = "ML_masks"
    out_path = os.path.join(root, folder_name)
    try:
        os.makedirs(out_path)
    except FileExistsError:
        pass  # directory already exists
    make_masks("Gphn", 2, out_path, image)
    make_masks("vGAT", 1, out_path, image)
    make_masks("gamma2", 3, out_path, image)
    make_soma_neuron_masks(out_path, image)

    make_intensity_images("Gphn", 2, out_path, confocal)
    make_intensity_images("vGAT", 1, out_path, confocal)
    make_intensity_images("gamma2", 3, out_path, confocal)

# make mask per channel
def make_masks(name, number, path, image):
    print("Generating masks...")

    out_file = os.path.join(path, image_name.replace('MaxZ_deconvolution.tif', name + '_labels.tif'))
    channel = image [:,:,number]
    cl_filename = "G134R_extended_data_code/models/" + name + '_object_model.cl'
    segmenter = apoc.ObjectSegmenter(opencl_filename=cl_filename)
    labels = segmenter.predict(channel)
    imsave(out_file, labels)


def make_soma_neuron_masks(path, image):
    # make some and neuron masks based on cell fill
    print("Generating soma and neuron masks...")
    channel = image [:,:,0]  # channel 0 is cell fill 
    median = median_filter(channel, size=20)
    segmented = cle.voronoi_otsu_labeling(median, spot_sigma=200, outline_sigma=30)
    segmented_excl_edges = clear_border(np.asarray(segmented))
    properties = measure.regionprops(segmented_excl_edges)
    statistics = {
    'area':       [p.area               for p in properties]
    }
    df = pd.DataFrame(statistics)
    size_threshold = np.max(df['area'])
    soma = cle.exclude_small_labels(segmented_excl_edges, maximum_size=size_threshold)
    lp = nsitk.laplacian_filter(median)
    std = nsitk.standard_deviation_filter(lp)
    li = threshold_li(std)
    neuron_li = std > li
    split_objects = neuron_li
    neuron_object = cle.connected_components_labeling_box(split_objects)
    properties_neuron = measure.regionprops(neuron_object)
    statistics_neuron = {
    'area':       [p.area               for p in properties_neuron]
    }
    df = pd.DataFrame(statistics_neuron)
    size_threshold_neuron = np.max(df['area'])
    neuron = cle.exclude_small_labels(neuron_object, maximum_size=size_threshold_neuron)
    out_path_soma = os.path.join(path, image_name.replace('MaxZ_deconvolution.tif', 'Soma_labels.tif'))
    imsave(out_path_soma, soma)
    
    out_path_neuron = os.path.join(path, image_name.replace('MaxZ_deconvolution.tif', 'Neuron_labels.tif'))
    imsave(out_path_neuron, neuron)


def make_intensity_images(name, number, path, image):
    print("Generating intensities...")

    out_file = os.path.join(path, confocal_name.replace('AvZ_confocal.tif', name + '_intensities.tif'))
    channel = image [:,:,number]
    background_rolling = rolling_ball(channel, radius=30)
    intensity = channel-background_rolling
    imsave(out_file, intensity)


for root, dirs, files in os.walk(image_folder):
    if image_name in files:
        execute(root)

print("...completed.")
