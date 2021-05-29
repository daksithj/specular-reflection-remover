import os
import random
import cv2
import glob
from tensorflow import keras
from DeepLearning.tools import translate_image, prepare_input, convert_image_to_hsv
from DatasetCreator.segment_dataset import extract_segments, get_patch
from DeepLearning.reflection_remover import get_diffuse
from PoseEstimation.pose_estimator import get_pose, get_disparity_map_matrix
from PoseEstimation.evaluate_pose import calculate_error, read_actual_locations

test_pairs = 3


def test_full():

    data_loc = 'Dataset/Test'
    data_num = 0

    image_1 = cv2.imread(f'{data_loc}/{data_num}_1.png')
    image_2 = cv2.imread(f'{data_loc}/{data_num}_2.png')

    output_1, output_2 = get_diffuse(image_1, image_2)

    disparity_map_matrix = get_disparity_map_matrix(data_loc, output_1.shape[:2])

    poses = get_pose(output_1, output_2, disparity_map_matrix, draw_line=True, draw_real=data_num)

    location_set = []

    for pose in poses:
        location_set.append(pose['location'])

    real_locations = read_actual_locations(data_loc, data_num)
    location_error = calculate_error(real_locations, location_set)

    print(f'Average distance error of all locations: {location_error}')


def generate_full_samples(model_name=None):

    if model_name is None:
        file_name = max(glob.iglob('Models/Generator/*.h5'), key=os.path.getctime)
    else:
        file_name = 'Models/Generator/' + model_name + '.h5'

    generator = keras.models.load_model(file_name)

    specular_dir = "Dataset/Specular/"
    data_num = len([name for name in os.listdir(specular_dir)
                    if os.path.isfile(os.path.join(specular_dir, name))]) // test_pairs

    data_num = random.randint(0, data_num)

    images = []

    for test in range(1, test_pairs + 1):
        file_name = specular_dir + str(data_num) + "_" + str(test) + ".png"
        image = cv2.imread(file_name)
        images.append(image)

    for image in images:

        segments = extract_segments(image)
        cv2.imshow("Frame", image)
        cv2.waitKey(0)
        seg = segments[0]
        patch = get_patch(image, seg)

        patch = convert_image_to_hsv(patch, specular=True, gray=False)
        patch = prepare_input(patch)

        patch_out = generator.predict(patch)

        patch_out = translate_image(patch_out)

        x1, x2, y1, y2 = seg

        image[x1:x2, y1:y2, :] = patch_out

        cv2.imshow("Frame", image)
        cv2.waitKey(0)


if __name__ == '__main__':
    test_full()
