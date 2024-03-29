import os
import struct
import numpy as np

"""
Slightly inspired by https://huggingface.co/datasets/mnist/blob/main/mnist.py,
an MIT licensed code for MNIST images reading and exporting.
"""

class MNIST_Reader():
    def __init__(self, dataset_category = "training", path = "data_files"):
        """
        Constructor function.
        Inputs:
            - dataset_category: training or testing samples to import.
            - path: relative path were the samples are located.
        Outputs:
            - None
        """

        self.dataset_category = dataset_category
        self.path = path
        self.read_images()

    def read_images(self,):
        """
        Python function for reading images from MINST and returning images belonging to
        cercain digits.
        Inputs:
            - Dataset category and path consumed when the object is defined.
        Outputs:
            - None. The function sets a dictionary were the keys are the digit labels and the 
              values are the image paths.
        """

        # Setting the filenames of the images and labels according to the dataset category
        if self.dataset_category == "training":
            filename_images = os.path.join(self.path, 'train-images')
            filename_labels = os.path.join(self.path, 'train-labels')
        elif self.dataset_category == "testing":
            filename_images = os.path.join(self.path, 'test-images')
            filename_labels = os.path.join(self.path, 'test-labels')
        else:
            raise NameError("dataset category should be 'testing' or 'training'")

        # Importing the image files per label
        with open(filename_labels, 'rb') as file_label:
            bits, num = struct.unpack(">II", file_label.read(8))
            digit_label = np.fromfile(file_label, dtype=np.int8)

        with open(filename_images, 'rb') as file_img:
            bits, num, rows, cols = struct.unpack(">IIII", file_img.read(16))
            single_img = np.fromfile(file_img, dtype=np.uint8).reshape(len(digit_label), rows, cols)
        
        # Creating the dictionary with digits as keys and image paths as values
        unique_labels = np.unique(digit_label)
        self.digits_images = {i:[] for i in unique_labels}
        for i in range(len(digit_label)):
            self.digits_images[digit_label[i]] += [single_img[i].astype(float)]

    def load_images_from_digit(self,digit=0):
        """
        Python function for loading the images according the the given digit.
        Inputs:
            - digit: image label representing the digit in the image.
        Outputs:
            - Returns training and testing sets based on a 50% split.
        """

        # Stack all imported images into a Numpy array
        samples = np.stack(self.digits_images[digit],axis=2)

        # Define the training and testing sets with a 50% split
        train_set = samples[:,:,:int(samples.shape[2]/2)]
        test_set = samples[:,:,int(samples.shape[2]/2):]

        # Returning the training and testing sets
        return train_set, test_set

    