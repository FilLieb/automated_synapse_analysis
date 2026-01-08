import os
from skimage.io import imread, imsave
import numpy as np
import pandas as pd 
from skimage import measure
import pyclesperanto_prototype as cle




# Set paths, everytime a file called Gphn_labels.tif is found, launch image processing analysis
image_folder = r"Z:/personal_data/Filip_Liebsch/SP8_Biocenter_small/2025-04-23_floxed/G134R_HET_project/"

# global variables
Gphn = "mScarlet-Gphn_labels.tif"
vGAT = "vGAT_labels.tif"
Gphn_wt = "mEGFP-Gphn_labels.tif"
neuron = "Neuron_labels.tif"
soma = "Soma_labels.tif"

Gphn_int = "mScarlet-Gphn_intensities.tif"
Gphn_wt_int = "mEGFP-Gphn_intensities.tif"
vGAT_int = "vGAT_intensities.tif"

df = pd.DataFrame()
save_path = ""

def execute(root, img, intensity_img):
    print(f"Found image in: {root}")
    image_path = os.path.join(root, img)
    image = imread(image_path)

    intensity_path = os.path.join(root, intensity_img)
    intensity_image = imread(intensity_path)

    folder_name = "measurements"
    out_path = os.path.join(root, folder_name)
    try:
        os.makedirs(out_path)
    except FileExistsError:
        pass  # directory already exists
    
    properties = measure.regionprops(image, intensity_image=intensity_image)
    name = img.split("_")[0]
    intensity_name = intensity_img.split("_")[0]
    statistics = {
    name + '_area':       [p.area               for p in properties],
    intensity_name + '_mean':       [p.mean_intensity     for p in properties]
    }
    global df
    df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame(statistics)], axis=1)
    global save_path
    save_path = out_path

def classify(root, img, inside):
    image_path = os.path.join(root, img)
    image = imread(image_path)

    inside_path = os.path.join(root, inside)
    inside_image = imread(inside_path)
    
    properties = measure.regionprops(image, intensity_image=inside_image)

    inside_name = inside.split("_")[0]

    statistics = {
    inside_name :       [p.mean_intensity     for p in properties]
    }

    for key in statistics:
        value = statistics[key]
        if isinstance(value, list):
            statistics[key] = [v > 0 for v in value]
        else:
            statistics[key] = value > 0

    global df
    df = pd.concat([df, pd.DataFrame(statistics)], axis=1)


def intensity(root, img, intensity_img):
    image_path = os.path.join(root, img)
    image = imread(image_path)

    intensity_path = os.path.join(root, intensity_img)
    intensity_image = imread(intensity_path)
    
    properties = measure.regionprops(image, intensity_image=intensity_image)

    name = img.split("_")[0]

    name_intensity = intensity_img.split("_")[0]

    statistics = {
    name + '_' + name_intensity  + '_mean':       [p.mean_intensity     for p in properties]
    }

    global df
    for key, value in statistics.items():
        if len(value) == 1 and len(df) > 0:
            statistics[key] = value * len(df)

    df = pd.concat([df, pd.DataFrame(statistics)], axis=1)

def size(root, img):
    image_path = os.path.join(root, img)
    image = imread(image_path)
    
    properties = measure.regionprops(image)

    name = img.split("_")[0]

    statistics = {
    name + '_area':       [p.area               for p in properties]
    }

    global df
    for key, value in statistics.items():
        if len(value) == 1 and len(df) > 0:
            statistics[key] = value * len(df)
    df = pd.concat([df, pd.DataFrame(statistics)], axis=1)

def parameters(root, img):
    image_path = os.path.join(root, img)
    image = imread(image_path)
    
    properties = cle.maximum_of_all_pixels(image)
   
    path_norm = os.path.normpath(root)
    path_list = path_norm.split(os.sep)

    statistics = {
    'Gphn_number':       [properties],
    'date':              [path_list[-5]],
    'condition':         [path_list[-4]],
    'cell':              [path_list[-3]]
    }

    global df
    for key, value in statistics.items():
        if len(value) == 1 and len(df) > 0:
            statistics[key] = value * len(df)
    df = pd.concat([df, pd.DataFrame(statistics)], axis=1)
    global save_path
    df.to_csv(os.path.join(save_path, path_list[-5] + "_cluster_analysis_" + path_list[-4] + "_" + path_list[-3] + ".tsv"), sep="\t")


def summarize(root, img):
    global df



    image_path = os.path.join(root, img)
   
    path_norm = os.path.normpath(root)
    path_list = path_norm.split(os.sep)
    global save_path
    df.to_csv(os.path.join(save_path, path_list[-5] + "_cluster_summary_" + path_list[-4] + "_" + path_list[-3] + ".tsv"), sep="\t")


for root, dirs, files in os.walk(image_folder):
    if Gphn in files:
        execute(root, Gphn_wt, Gphn_wt_int)
        classify(root, Gphn_wt, vGAT)
        classify(root, Gphn_wt, Gphn_int)
        classify(root, Gphn_wt, neuron)
        classify(root, Gphn_wt, soma)
        intensity(root, neuron, Gphn_wt_int)
        intensity(root, soma, Gphn_wt_int)
        intensity(root, neuron, Gphn_int)
        intensity(root, soma, Gphn_int)
        size(root, neuron)
        size(root, soma)
        parameters(root, Gphn_wt)

print("...completed.")