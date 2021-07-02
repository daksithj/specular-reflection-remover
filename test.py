from DeepLearning.gan import SpecToPoseNet
from DeepLearning.dataset_gan import ImageDataSet
from PoseEstimation.pose_estimator import get_pose, get_disparity_map_matrix
from PoseEstimation.evaluate_pose import calculate_error
from SingleImage.single_image import single_image_removal
from MultiImage.MultiImageModelBasedApproach import multi_image_removal
from skimage import metrics
import numpy as np
import time
import cv2

test_pairs = 3


def test_full():

    dataset = ImageDataSet('Test')

    network = SpecToPoseNet('model_1', dataset)

    images, targets, real_locations = dataset.get_test_pair()

    image_1, image_2 = images

    output_1, output_2 = network.get_output([image_1, image_2])

    disparity_map_matrix = get_disparity_map_matrix(dataset.matrix_dir, output_1.shape[:2])

    poses = get_pose( output_1, output_2, disparity_map_matrix, draw_line=True, real_locations=real_locations)

    location_set = []

    for pose in poses:
        location_set.append(pose['location'])

    location_error, fail_rate = calculate_error(real_locations, location_set)

    print(f'Average distance error of all locations: {location_error}')


def test_script():

    mean_squared_error = 0

    peak_signal_noise_ratio = 0

    structural_similarity = 0

    object_loss = 0

    pose_loss = 0

    fail_rate = 0

    dataset = ImageDataSet('Test')

    network = SpecToPoseNet('model_1', dataset)

    iterations = 100

    for i in range(iterations):

        images, targets, real_locations = dataset.get_test_pair()

        image_1, image_2 = images

        target_1, target_2 = targets

        start_time = time.time()

        output_1, output_2 = network.get_output([image_1, image_2])

        print("--- %s seconds ---" % (time.time() - start_time))

        mse = 0.5 * (metrics.mean_squared_error(output_1, target_1) + metrics.mean_squared_error(output_2, target_2))

        mean_squared_error += mse

        peak = 0.5 * (metrics.peak_signal_noise_ratio(output_1, target_1) +
                      metrics.peak_signal_noise_ratio(output_2, target_2))

        peak_signal_noise_ratio += peak

        ss = 0.5 * (metrics. structural_similarity(output_1, target_1, multichannel=True) +
                    metrics. structural_similarity(output_2, target_2, multichannel=True))

        structural_similarity += ss

        disparity_map_matrix = get_disparity_map_matrix(dataset.matrix_dir, output_1.shape[:2])

        poses = get_pose(output_1, output_2, disparity_map_matrix)

        location_set = []

        for pose in poses:
            location_set.append(pose['location'])

        ob_loss = np.abs((len(real_locations) - len(location_set)))
        location_error, fail_error = calculate_error(real_locations, location_set)

        object_loss += ob_loss
        pose_loss += location_error
        fail_rate += fail_error

        f = open("final_model.txt", "a")
        f.write(f"{i},{mse},{ss},{peak},{ob_loss},{location_error},{fail_error}\n")
        f.close()

    print(mean_squared_error/iterations)
    print(peak_signal_noise_ratio / iterations)
    print(structural_similarity / iterations)
    print(object_loss / iterations)
    print(pose_loss / iterations)
    print(fail_rate / 6 * iterations)


def test_single():
    dataset = ImageDataSet('Test')

    images, targets, real_locations = dataset.get_test_pair()

    image_1, image_2 = images

    target_1, target_2 = targets

    output_1, _ = single_image_removal(image_1)
    output_2, _ = single_image_removal(image_2)

    output_1 = cv2.medianBlur(output_1, 1)
    output_2 = cv2.medianBlur(output_2, 1)

    mse = 0.5 * (metrics.mean_squared_error(output_1, target_1) + metrics.mean_squared_error(output_2, target_2))

    peak = 0.5 * (metrics.peak_signal_noise_ratio(output_1, target_1) +
                  metrics.peak_signal_noise_ratio(output_2, target_2))

    ss = 0.5 * (metrics.structural_similarity(output_1, target_1, multichannel=True) +
                metrics.structural_similarity(output_2, target_2, multichannel=True))

    print(mse)
    print(peak)
    print(ss)

    disparity_map_matrix = get_disparity_map_matrix(dataset.matrix_dir, output_1.shape[:2])

    poses = get_pose(output_1, output_2, disparity_map_matrix, draw_line=True)

    location_set = []

    for pose in poses:
        location_set.append(pose['location'])

    location_error, fail_rate = calculate_error(real_locations, location_set)

    print(f'Average distance error of all locations: {location_error}')


def test_multi():
    dataset = ImageDataSet('Test')

    images, targets, real_locations = dataset.get_test_pair()

    image_1, image_2 = images

    image_1 = cv2.imread("4pair.JPG")
    image_2 = cv2.imread("4pair2.JPG")
    image_3 = cv2.imread("4pair2.JPG")

    output_1, spec_1, output_2, spec_2, output_3, spec_3 = multi_image_removal(image_1, image_2, image_3)

    cv2.imshow("window", output_1)
    cv2.waitKey(0)

    cv2.imshow("window", output_2)
    cv2.waitKey(0)

    cv2.imshow("window", output_3)
    cv2.waitKey(0)

    disparity_map_matrix = get_disparity_map_matrix(dataset.matrix_dir, output_1.shape[:2])

    poses = get_pose(output_1, output_2, disparity_map_matrix, draw_line=True, real_locations=real_locations)

    location_set = []

    for pose in poses:
        location_set.append(pose['location'])

    location_error, fail_rate = calculate_error(real_locations, location_set)

    print(f'Average distance error of all locations: {location_error}')


if __name__ == '__main__':
    test_full()
