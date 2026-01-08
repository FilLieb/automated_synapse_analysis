import os
from skimage.io import imread
import matplotlib.pyplot as plt

#here you can define a subfolder such as a channel or staining for training classifier
channel = "vGAT"

image_folder = "G134R_extended_data_code/2_train_object_models/training_data/" + channel + "/images/"
print("Image folder path:", image_folder)

masks_folder = "G134R_extended_data_code/2_train_object_models/training_data/" + channel + "/masks/"
print("Masks folder path:", masks_folder)

#display one image and its corresponding mask from the folder to verify paths are correct
image_path = os.path.join(image_folder, os.listdir(image_folder)[0])
image = imread(image_path)

masks_path = os.path.join(masks_folder, os.listdir(masks_folder)[0])
masks = imread(masks_path)


f, a = plt.subplots(1,3, figsize=(15,5))
a[0].imshow(image, cmap='gray')
a[1].imshow(masks, vmin=0, vmax=2)
a[2].imshow(image, cmap='gray')
a[2].contour(masks, colors='r', linewidths=0.5)

plt.show()