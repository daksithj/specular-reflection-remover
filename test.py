from DeepLearning.gan import SpecToPoseNet
from DeepLearning.dataset_gan import ImageDataSet
from PoseEstimation.pose_estimator import get_pose, get_disparity_map_matrix
from PoseEstimation.evaluate_pose import calculate_error

test_pairs = 3


def test_full():

    dataset = ImageDataSet('Test')

    network = SpecToPoseNet('model_1', dataset)

    images, targets, real_locations = dataset.get_test_pair()

    image_1, image_2 = images

    output_1, output_2 = network.get_output([image_1, image_2])

    disparity_map_matrix = get_disparity_map_matrix(dataset.matrix_dir, output_1.shape[:2])

    poses = get_pose(output_1, output_2, disparity_map_matrix, draw_line=True, real_locations=real_locations)

    location_set = []

    for pose in poses:
        location_set.append(pose['location'])

    location_error = calculate_error(real_locations, location_set)

    print(f'Average distance error of all locations: {location_error}')


if __name__ == '__main__':
    test_full()
