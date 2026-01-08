import numpy as np
import pandas as pd 
import os
import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt

from skimage import measure
from skimage.io import imread, imsave
from skimage.segmentation import clear_border
from skimage.filters import threshold_li

from scipy.ndimage import median_filter
import napari_simpleitk_image_processing as nsitk





image_folder = r"Z:/personal_data/Filip_Liebsch/SP8_Biocenter_small/2025-04-23_floxed/S325D_project/data/2025-03-31/WT/cell1/Z-projection"
image_name = "MaxZ_deconvolution.tif"
image_path = os.path.join(image_folder, image_name)
image = imread(image_path)

channel = image [:,:,0]  # channel 0 is cell fill 
median = median_filter(channel, size=20)

segmented = cle.voronoi_otsu_labeling(median, spot_sigma=200, outline_sigma=30)
segmented_excl_edges = clear_border(np.asarray(segmented))

properties = measure.regionprops(segmented_excl_edges)
statistics = {
    'area':       [p.area               for p in properties]
}

df = pd.DataFrame(statistics)
# measure max area
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
# measure max area
size_threshold_neuron = np.max(df['area'])

neuron = cle.exclude_small_labels(neuron_object, maximum_size=size_threshold_neuron)

out_path_soma = os.path.join(image_folder, image_name.replace('MaxZ_deconvolution.tif', 'Soma_labels.tif'))
imsave(out_path_soma, soma)

out_path_neuron = os.path.join(image_folder, image_name.replace('MaxZ_deconvolution.tif', 'Neuron_labels.tif'))
imsave(out_path_neuron, neuron)


print("...completed.")
