from tensorflow.keras.utils import Sequence
from DeepLearning.tools import convert_image_to_hsv
import os
import random
import cv2
import numpy as np

num_objects = 6
image_size = 256
pairs = 2

specular_dir = "Dataset/Specular/"
diffuse_dir = "Dataset/Diffuse/"

image_extension = '.png'


class ImageDataSet(Sequence):

    def __init__(self, batch_size=1):

        self.batch_size = batch_size
        self.specular_dir = specular_dir
        self.diffuse_dir = diffuse_dir

        self.grayscale = 0

        self.data_num = len([name for name in os.listdir(self.specular_dir)
                             if os.path.isfile(os.path.join(self.specular_dir, name))]) // pairs

        self.sequence = list(range(0, self.data_num))
        random.shuffle(self.sequence)

    def __len__(self):
        return self.data_num / self.batch_size

    def __getitem__(self, idx):

        # If batch size exceeds the remaining data size
        if idx + self.batch_size > self.data_num:
            size = self.data_num - idx
        else:
            size = self.batch_size

        im_range = self.sequence[idx: idx + size]

        if self.grayscale == 1:
            cv_flag = cv2.IMREAD_GRAYSCALE
        else:
            cv_flag = cv2.IMREAD_COLOR

        s_views = []

        d_views = []
        for x in im_range:
            s_angle = []
            for y in range(1, pairs + 1):

                s_view_1_name = self.specular_dir + str(x) + "_" + str(y) + image_extension
                s_view_1 = cv2.imread(s_view_1_name, flags=cv_flag)
                s_view_1 = convert_image_to_hsv(s_view_1, gray=False, specular=True)
                s_view_1 = s_view_1 / 127.5 - 1.0
                s_angle.append(s_view_1)
            s_image = np.concatenate(s_angle, axis=1)
            s_views.append(s_image)

            d_angle = []
            for y in range(1, pairs + 1):
                d_view_1_name = self.diffuse_dir + str(x) + "_" + str(y) + image_extension
                d_view_1 = cv2.imread(d_view_1_name, flags=cv_flag)
                d_view_1 = convert_image_to_hsv(d_view_1, gray=False, specular=False)
                d_view_1 = d_view_1 / 127.5 - 1.0
                d_angle.append(d_view_1)
            d_image = np.concatenate(d_angle, axis=1)
            d_views.append(d_image)

        return np.asarray(s_views), np.asarray(d_views)

    def on_epoch_end(self):
        random.shuffle(self.sequence)
