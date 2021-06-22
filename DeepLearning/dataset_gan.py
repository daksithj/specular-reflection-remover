from tensorflow.keras.utils import Sequence
import os
import glob
import random
import cv2
import numpy as np
import pickle as pkl

dataset_folder_name = "Output"

image_extension = '.png'
location_file_prefix = 'location_'
location_extension = ""


class ImageDataSet(Sequence):

    def __init__(self, dataset_name, batch_size=1):

        self.batch_size = batch_size

        self.dataset_name = dataset_name

        self.dataset_folder = self.check_dataset_folders(dataset_name)
        self.specular_dir = self.dataset_folder + "/Specular/"
        self.diffuse_dir = self.dataset_folder + "/Diffuse/"
        self.location_dir = self.dataset_folder + "/Locations/"
        self.matrix_dir = self.dataset_folder + "/Matrix"

        self.data_num, self.pairs, self.image_size, self.channels = self.check_dataset_integrity(self.dataset_folder)

        self.grayscale = 0

        self.sequence = list(range(0, self.data_num))
        random.shuffle(self.sequence)

    @staticmethod
    def check_dataset_folders(dataset_name):

        dataset_folder = f'DatasetCreator/{dataset_folder_name}/{dataset_name}'

        if not os.path.exists(dataset_folder):
            raise ImportError("Cannot find dataset folder.")

        if not os.path.exists(dataset_folder + '/Specular'):
            raise ImportError("Cannot find specular data folder.")

        if not os.path.exists(dataset_folder + '/Diffuse'):
            raise ImportError("Cannot find diffuse data folder.")

        if not os.path.exists(dataset_folder + '/Matrix'):
            raise ImportError("Cannot find matrix data folder.")

        if not os.path.exists(dataset_folder + '/Locations'):
            raise ImportError("Cannot find location data folder.")

        return dataset_folder

    @staticmethod
    def check_dataset_integrity(dataset_folder):

        specular_folder = dataset_folder + '/Specular/'
        diffuse_folder = dataset_folder + '/Diffuse/'
        matrix_folder = dataset_folder + '/Matrix/'
        location_folder = dataset_folder + '/Locations/'

        specular_files = set(map(os.path.basename, glob.glob(f'{specular_folder}*.png')))
        diffuse_files = set(map(os.path.basename, glob.glob(f'{diffuse_folder}*.png')))

        # Check for file integrity between specular and diffuse
        if len(specular_files ^ diffuse_files) > 0:
            raise ImportError("All files or file names in specular and diffuse folders do not match")

        # Check for file integrity in pairs
        specular_files = os.listdir(specular_folder)
        index_list = []
        for file in specular_files:
            file = file.replace(".", "_")
            index_list.append(int(file.split("_")[1]))

        index_list = np.asarray(index_list)
        pairs = np.max(index_list)

        pair_count = np.bincount(index_list)[1:]

        data_num = np.max(pair_count)

        if not data_num == np.min(pair_count):
            raise ImportError("Mismatch of pairs in the dataset")

        test_image = cv2.imread(specular_folder + specular_files[0])

        image_size = test_image.shape[0]
        channels = test_image.shape[2]

        # Check matrix folder
        matrix_files_required = ['K_matrix_0_1', 'K_matrix_0_2', 'RT_matrix_0_1', 'RT_matrix_0_2']
        matrix_files = os.listdir(matrix_folder)

        if not (set(matrix_files) == set(matrix_files_required)):
            raise ImportError("Required matrix files missing")

        # Check location folder
        for data in range(data_num):
            if not os.path.exists(f'{location_folder}{location_file_prefix}{data}'):
                raise ImportError("A location file is missing")

        return data_num, pairs, image_size, channels

    @staticmethod
    def check_image_dimensions(specular_folder, diffuse_folder):
        specular_files = os.listdir(specular_folder)

        test_image = cv2.imread(specular_folder + specular_files[0])

        image_size = test_image.shape[0]
        channels = test_image.shape[2]

        if not test_image.shape[1] == image_size:
            raise ImportError("All images must have a 1:1 aspect ratio")

        # Check if size is a power of two
        if not (image_size & (image_size - 1) == 0) and image_size != 0:
            raise ImportError("Image size must be a power of 2")

        for file in specular_files:
            read = cv2.imread(specular_folder + file)
            if not ((read.shape[0] == image_size) and (read.shape[1] == image_size) and (read.shape[2] == channels)):
                raise ImportError("All images in the specular folder must be in the same dimension")

        diffuse_files = os.listdir(diffuse_folder)

        for file in diffuse_files:
            read = cv2.imread(diffuse_folder + file)
            if not ((read.shape[0] == image_size) and (read.shape[1] == image_size) and (read.shape[2] == channels)):
                raise ImportError("All images in the diffuse folder must be in the same dimension")

    @staticmethod
    def convert_image_to_hsv(image, gray=False, specular=True):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        if specular:
            clip_limit = 6.0
            title_grid_size = (6, 6)
        else:
            clip_limit = 2.0
            title_grid_size = (3, 3)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=title_grid_size)
        v = clahe.apply(v)

        hsv_image = cv2.merge([h, s, v])
        hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        if specular:
            gray_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
            thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)[1]

            hsv_image = cv2.inpaint(hsv_image, thresh, 3, cv2.INPAINT_TELEA)
        if gray:
            hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2GRAY)
            hsv_image = np.array(hsv_image)
            hsv_image = np.expand_dims(hsv_image, axis=-1)
        else:
            hsv_image = np.array(hsv_image)

        return hsv_image

    @staticmethod
    def prepare_input(image, expand=False):
        image = image / 127.5 - 1.0
        image = np.asarray(image)
        if expand:
            image = np.expand_dims(image, axis=0)
        return image

    @staticmethod
    def translate_image(image):

        image = 0.5 * image + 0.5
        image = image * 255
        image = np.asarray(image).astype(np.uint8)
        image = np.squeeze(image, axis=0)

        return image

    def get_test_pair(self):

        x = random.randint(0, self.data_num)

        images = []

        for y in range(1, self.pairs + 1):
            s_view_name = self.specular_dir + str(x) + "_" + str(y) + image_extension
            s_view = cv2.imread(s_view_name)
            images.append(s_view)

        loc_file = open(f'{self.location_dir}/location_{x}', 'rb')
        locations = pkl.load(loc_file)
        loc_file.close()

        return images, locations

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

        locations = []

        for x in im_range:
            s_angle = []
            for y in range(1, self.pairs + 1):

                s_view_1_name = self.specular_dir + str(x) + "_" + str(y) + image_extension
                s_view_1 = cv2.imread(s_view_1_name, flags=cv_flag)
                s_view_1 = self.convert_image_to_hsv(s_view_1, gray=False, specular=True)
                s_view_1 = self.prepare_input(s_view_1)
                s_angle.append(s_view_1)
            s_image = np.concatenate(s_angle, axis=1)
            s_views.append(s_image)

            d_angle = []
            for y in range(1, self.pairs + 1):
                d_view_1_name = self.diffuse_dir + str(x) + "_" + str(y) + image_extension
                d_view_1 = cv2.imread(d_view_1_name, flags=cv_flag)
                d_view_1 = self.convert_image_to_hsv(d_view_1, gray=False, specular=False)
                d_view_1 = self.prepare_input(d_view_1)
                d_angle.append(d_view_1)
            d_image = np.concatenate(d_angle, axis=1)
            d_views.append(d_image)

            location_file = open(self.location_dir + location_file_prefix + str(x) + location_extension, 'rb')
            obj_locations = pkl.load(location_file)
            locations.append(obj_locations)
            location_file.close()

        return np.asarray(s_views), [np.asarray(d_views), locations]

    def on_epoch_end(self):
        random.shuffle(self.sequence)
