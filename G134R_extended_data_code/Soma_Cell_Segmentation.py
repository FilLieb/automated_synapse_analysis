import numpy as np
import pandas as pd 
import os
import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt
from skimage import measure
from skimage.io import imread, imsave
from skimage.filters import difference_of_gaussians
from skimage.segmentation import clear_border
from scipy.ndimage import median_filter , gaussian_filter, maximum_filter
from napari_segment_blobs_and_things_with_membranes import seeded_watershed


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
print("size threshold for soma:", size_threshold)

soma = cle.exclude_small_labels(segmented_excl_edges, maximum_size=size_threshold)

# create a new plot
fig, axes = plt.subplots(1,1)
axes.imshow(channel, cmap=plt.cm.gray)
axes.contour(soma, [0.5], linewidths=1.2, colors='r')
plt.show()


#sigma_spot_detection = 20
#sigma_outline = 1
#blurred = cle.gaussian_blur(channel, sigma_x=sigma_spot_detection, sigma_y=sigma_spot_detection, sigma_z=sigma_spot_detection)

#detected_spots = cle.detect_maxima_box(blurred, radius_x=sigma_outline, radius_y=sigma_outline, radius_z=sigma_outline)

#number_of_spots = cle.sum_of_all_pixels(detected_spots)
#print("number of detected spots", number_of_spots)






#blobs_sobel = cle.sobel(median)
#dog = difference_of_gaussians(blobs_sobel, 2, 30)
#segmented = cle.gauss_otsu_labeling(dog, outline_sigma=2)

#cle.imshow(segmented)

#blurred = maximum_filter(segmented, size=5)


#neuron = seeded_watershed(blurred, soma)
#cle.imshow(neuron, labels=True)




# edges = cle.standard_deviation_box(median, radius_x=25, radius_y=25, radius_z=25)




#stackview.imshow(edges)

#dog = difference_of_gaussians(median, 2, 30)


#intensity_equivalized = cle.divide_by_gaussian_background(channel, sigma_x=100, sigma_y=100)
#cle.imshow(intensity_equivalized, title="Intensity Equalized")





# out_path = os.path.join(image_folder, image_name.replace('MaxZ_deconvolution.tif', 'Cell_labels.tif'))

# imsave(out_path, labels)

print("...completed.")
