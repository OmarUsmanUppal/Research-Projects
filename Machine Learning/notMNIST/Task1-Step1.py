import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
from IPython.display import display, Image
import imageio as im
# import png

def subfolders_extract(foldername, force=False):
    data_folders=[]
    for class_folder in os.listdir(foldername):
        if os.path.isdir(os.path.join(foldername, class_folder)):
            data_folders.append (os.path.join(foldername, class_folder))

    # data_folders = [os.path.join(filename, d) for d in sorted(os.listdir(filename))if os.path.isdir(os.path.join(filename, d))]
    print(data_folders)
    return data_folders

train_dirname = "notMNIST_large"
test_dirname =  "notMNIST_small"
train_folders = subfolders_extract(train_dirname)
test_folders = subfolders_extract(test_dirname)
print(test_folders)
print(type(train_folders))
image_size=28
pixel_depth=255

def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    # images_files = image_files[20000:]
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print("folder name",folder, "dataset", dataset)
    num_images = 0
    for image in image_files:
        print(num_images)
        image_file = os.path.join(folder, image)
        try:
            image_data = (im.imread(image_file).astype(float) ) / pixel_depth

#             print("image size", image_data.shape)
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))

    # print('Full dataset tensor:', dataset.shape)
    # print('Mean:', np.mean(dataset))
    # print('Standard deviation:', np.std(dataset))
    return dataset

# train_datasets = load_letter(train_folders[0], 45000)
# for i in range(9):
# print(train_folders[0])
# train_datasets = load_letter(train_folders[3], 1800)
# print("NEEEEEEEEEEEEEEEXXXXXXXXXXXXTTTTTTTTTT1")
# train_datasets = load_letter(train_folders[1], 1800)
# print("NEEEEEEEEEEEEEEEXXXXXXXXXXXXTTTTTTTTTT2")
#
test_datasets = load_letter(test_folders[0], 1800)



