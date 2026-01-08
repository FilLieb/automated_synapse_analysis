import apoc

# after several rounds of optimization this script was used to train the Gphn object model
# optimization was performed to select narrow down important features for making the pixel classification decision 

#here you can define a subfolder such as a channel or staining for training classifier
channel = "Gphn"

image_folder = "G134R_extended_data_code/2_train_object_models/training_data/" + channel + "/images/"
masks_folder = "G134R_extended_data_code/2_train_object_models/training_data/" + channel + "/masks/"

# this is where the model will be saved
cl_filename = "G134R_extended_data_code/models/" + channel + '_object_model.cl'
apoc.erase_classifier(cl_filename) # delete it if it was existing before
# setup classifier and where it should be saved
segmenter = apoc.ObjectSegmenter(opencl_filename=cl_filename,
                                 max_depth=5,
                                 num_ensembles=1000)

# setup feature set used for training

features = "difference_of_gaussian=5 laplace_box_of_gaussian_blur=5"

# train classifier on folders
apoc.erase_classifier(cl_filename)
apoc.train_classifier_from_image_folders(
    segmenter,
    features,
    image = image_folder,
    ground_truth = masks_folder)

print("Training completed and model saved to " + cl_filename)
