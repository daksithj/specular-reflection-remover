import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import math
from random import randint

from PoseEstimation.object_detection import get_object_masks
from PoseEstimation.evaluate_pose import get_closest_location


def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):

        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        image = cv2.pyrDown(image, dstsize=(col // 2, row // 2))
    return image


# Get the disparity map of the scene
def get_disparity_map(img_1, img_2, down_sample=0):

    # Downsample each image if required
    img_1_downsampled = downsample_image(img_1, down_sample)
    img_2_downsampled = downsample_image(img_2, down_sample)

    # Set disparity parameters
    win_size = 3
    min_disp = -1
    max_disp = 31
    num_disp = max_disp - min_disp  # Needs to be divisible by 16

    # Create Block matching object.
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=11,
                                   uniquenessRatio=5,
                                   speckleWindowSize=5,
                                   speckleRange=5,
                                   disp12MaxDiff=2,
                                   P1=8 * 3 * win_size ** 2,
                                   P2=32 * 3 * win_size ** 2)

    disparity_map = stereo.compute(img_1_downsampled, img_2_downsampled)

    return disparity_map


def get_disparity_map_matrix(folder, img_shape):

    # Reading Camera calibrations
    k_file = open(f'{folder}/K_matrix_0_1', 'rb')
    k_1 = pkl.load(k_file)
    k_file.close()

    k_file = open(f'{folder}/K_matrix_0_2', 'rb')
    k_2 = pkl.load(k_file)
    k_file.close()

    # Reading rotation and translation matrices for camera 1
    rt_file = open(f'{folder}/RT_matrix_0_1', 'rb')
    rt_1 = pkl.load(rt_file)

    r_1 = np.zeros((4, 4))
    r_1[:3, :3] = rt_1[:, :3]
    r_1[3][3] = 1

    t_1 = np.zeros((4, 4))
    t_1[:3, 3:] = rt_1[:, 3:]
    t_1[:3, :3] = np.identity(3)
    t_1[3][3] = 1

    rt_file.close()

    # Reading rotation and translation matrices for camera 2
    rt_file = open(f'{folder}/RT_matrix_0_2', 'rb')
    rt_2 = pkl.load(rt_file)

    r_2 = np.zeros((4, 4))
    r_2[:3, :3] = rt_2[:, :3]
    r_2[3][3] = 1

    t_2 = np.zeros((4, 4))
    t_2[:3, 3:] = rt_2[:, 3:]
    t_2[:3, :3] = np.identity(3)
    t_2[3][3] = 1

    rt_file.close()

    tolerance = 0.01

    if not (abs(r_1 - r_2) <= tolerance).all():
        raise Exception("Relative rotation found")

    translation = np.zeros(3)
    disp_found = False

    for x in range(3):
        translation[x] = t_1[x][3] - t_2[x][3]
        if translation[x] > tolerance:
            if disp_found:
                raise Exception("Multiple axes displacement found")
            disp_found = True

    if not disp_found:
        raise Exception("Camera translation is same")

    rotation = np.identity(3)

    disparity_map_matrix = np.zeros((4, 4))

    cv2.stereoRectify(cameraMatrix1=k_1, cameraMatrix2=k_2,
                      distCoeffs1=0, distCoeffs2=0,
                      imageSize=img_shape, R=rotation, T=translation,
                      R1=None, R2=None, P1=None, P2=None,
                      Q=disparity_map_matrix, alpha=1)

    return disparity_map_matrix


def get_3d_points(disparity_map, disparity_map_matrix):

    disparity_map = disparity_map / 16.0

    focal_length = disparity_map_matrix[2][3]

    displacement = -disparity_map_matrix[3][2]

    principle_point_x = disparity_map_matrix[0][3]
    principle_point_y = disparity_map_matrix[1][3]

    location_map = np.zeros(shape=(disparity_map.shape[0], disparity_map.shape[1], 3))
    lambda_map = np.zeros(shape=disparity_map.shape)

    # https://answers.opencv.org/question/187734/derivation-for-perspective-transformation-matrix-q/
    for a in range(disparity_map.shape[0]):
        for b in range(disparity_map.shape[1]):
            disp = disparity_map[a][b]
            if disp <= 0:
                disp = -1

            w = disp * displacement
            z = focal_length/w
            y = (b + principle_point_y)/w
            x = (a + principle_point_x)/w

            location_map[a][b][0] = x
            location_map[a][b][1] = y
            location_map[a][b][2] = z

            lambda_map[a][b] = w

    return location_map, lambda_map, disparity_map_matrix


def remove_outliers(location_map, depth_limit=100):

    data = location_map[:, :, 2]
    data[data < 0] = 0
    data[data > depth_limit] = 0


# Apply mask and separate out objects in the location set
def apply_mask(location_map, object_masks):

    object_points = []
    for mask in object_masks:
        mask = np.repeat(mask.reshape(mask.shape[0], mask.shape[1], 1), 3, axis=2)
        point = location_map * mask
        object_points.append(point)

    return object_points


def project_2d_point(x, y, w, disparity_map_matrix):

    principle_point_x = disparity_map_matrix[0][3]
    principle_point_y = disparity_map_matrix[1][3]

    c = int((x * w) - principle_point_y)
    d = int((y * w) - principle_point_x)

    return c, d


def get_image_coordinates(coordinates, lambda_map, object_point, disparity_map_matrix):

    x, y, z = coordinates
    x_array = object_point[:, :, 0]

    idx = (np.abs(x_array - x)).argmin()
    x_1, x_2 = np.unravel_index(idx, x_array.shape)

    y_array = object_point[:, :, 1]

    idx = (np.abs(y_array - y)).argmin()
    y_1, y_2 = np.unravel_index(idx, y_array.shape)

    z_array = object_point[:, :, 2]

    idx = (np.abs(z_array - z)).argmin()
    z_1, z_2 = np.unravel_index(idx, z_array.shape)

    w = np.mean([lambda_map[x_1][x_2], lambda_map[y_1][y_2], lambda_map[z_1][z_2]])

    if w <= 0:
        c, d = 0, 0
    else:
        c, d = project_2d_point(y, x, w, disparity_map_matrix)
    return c, d


# def angle_between_lines(point_1, point_2, point_3):
#
#     ba = point_1 - point_2
#     bc = point_3 - point_2
#
#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#
#     return cosine_angle


def get_axis(object_point):

    x_points = object_point[:, :, 0]
    x_points = x_points[np.nonzero(x_points)]
    x_mid = np.median(x_points)

    y_points = object_point[:, :, 1]
    y_points = y_points[np.nonzero(y_points)]
    y_mid = np.median(y_points)

    z_points = object_point[:, :, 2]
    z_points = z_points[np.nonzero(z_points)]
    z_mid = np.median(z_points)

    coordinate = np.asarray([x_mid, y_mid, z_mid])

    corner = np.zeros(3)
    max_dist = 0
    for a in range(object_point.shape[0]):
        for b in range(object_point.shape[1]):

            if object_point[a][b][2] == 0:
                continue

            point = object_point[a][b][:2]
            z_point = object_point[a][b][2]
            point = np.append(point, z_mid)
            squared_dist = np.sum((point - coordinate) ** 2, axis=0)
            dist = np.sqrt(squared_dist)

            if dist > max_dist:
                max_dist = dist
                corner = point
                corner[2] = z_point

    dif = coordinate - corner

    corner_2 = coordinate + dif

    return coordinate, corner, corner_2


# Get pitch and yaw based on
# https://stackoverflow.com/questions/58469297/how-do-i-calculate-the-yaw-pitch-and-roll-of-a-point-in-3d
def get_object_rotation(point_1, point_2):

    if point_1[0] > point_2[0]:
        point_1, point_2 = point_2, point_1

    dx = point_1[0] - point_2[0]
    dy = point_1[1] - point_2[1]
    dz = point_1[2] - point_2[2]

    # Rotation of the XZ plane (angle from x towards z axis)
    yaw = math.atan2(dz, dx)

    # Imagine X-z is a 2D plane. After the above yaw rotation the length of the line in that plane
    # is math.sqrt(dx*dx + dz*dz) like euclidean distance.
    # pitch is the angle between the above line and the actual line. A triangle can be formed with the
    # flattened line, the y direction height and the actual line
    pitch = math.atan2(dy, math.sqrt(dx*dx + dz*dz))

    return np.asarray([yaw, pitch])


# Mark each object in the image. Alpha is the transparency (lower is more transparent)
def draw_overlay(img, object_point, colour=None, alpha=0.8):

    if colour is None:
        colour = (randint(0, 255), randint(0, 255), randint(0, 255))

    for y in range(object_point.shape[1]):
        for x in range(object_point.shape[0]):
            if object_point[x][y][2] == 0:
                continue
            overlay = img.copy()
            cv2.circle(overlay, (y, x), 1, colour, -1)  # A filled rectangle
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    return img


# Draw the axis of each object in the image
def draw_axis(img, object_point, centre, axis_point_1, lambda_map, disparity_map_matrix, colour=None):

    if colour is None:
        colour = (255, 255, 255)

    corner_1_x, corner_1_y = get_image_coordinates(axis_point_1, lambda_map, object_point, disparity_map_matrix)
    centre_x, centre_y = get_image_coordinates(centre, lambda_map, object_point, disparity_map_matrix)

    dif_x = centre_x - corner_1_x
    dif_y = centre_y - corner_1_y

    corner_2_x = centre_x + dif_x
    corner_2_y = centre_y + dif_y

    line_array = np.array([(corner_1_x, corner_1_y), (centre_x, centre_y), (corner_2_x, corner_2_y)])

    cv2.drawContours(img, [line_array], 0, colour, 2)

    cv2.circle(img, (centre_x, centre_y), 4, colour, -1)


def draw_real_locations(img, object_point, centre, lambda_map, disparity_map_matrix, real_locations):

    closest, _ = get_closest_location(real_locations, centre)

    centre_x, centre_y = get_image_coordinates(closest, lambda_map, object_point, disparity_map_matrix)

    cv2.circle(img, (centre_x, centre_y), 4, (0, 0, 255), -1)


# Plot and show the the depth
def show_depth_maps(object_points):

    for item in object_points:
        plt.imshow(item[:, :, 2], 'magma')
        plt.show()


def show_object_masks(object_masks):

    for mask in object_masks:
        cv2.imshow('Mask', mask)
        cv2.waitKey(0)


def get_pose(img_1, img_2, disparity_map_matrix, draw_line=False, draw_points=False, real_locations=None,
             get_image=False):

    object_masks = get_object_masks(img_1, k_1=15, k_2=3)
    disparity_map = get_disparity_map(img_1, img_2)

    location_map, lambda_map, disparity_map_matrix = get_3d_points(disparity_map, disparity_map_matrix)
    remove_outliers(location_map)

    object_points = apply_mask(location_map, object_masks)

    poses = []
    for item in object_points:
        centre, axis_point_1, axis_point_2 = get_axis(item)
        rotation = get_object_rotation(axis_point_1, axis_point_2)

        poses.append({'location': centre, 'rotation': rotation})
        colour = (randint(0, 255), randint(0, 255), randint(0, 255))

        if draw_points:
            img_1 = draw_overlay(img_1, item, colour=colour, alpha=0.07)

        if draw_line:
            draw_axis(img_1, item, centre, axis_point_1, lambda_map, disparity_map_matrix, colour=colour)

        if real_locations is not None:
            draw_real_locations(img_1, item, centre, lambda_map, disparity_map_matrix, real_locations)

    if get_image:
        return img_1

    if draw_points or draw_line:
        cv2.imshow('Window', img_1)
        cv2.waitKey(0)

    return poses


def get_pose_with_centres(img_1, img_2, disparity_map_matrix):

    object_masks = get_object_masks(img_1, k_1=15, k_2=3)
    disparity_map = get_disparity_map(img_1, img_2)

    location_map, lambda_map, disparity_map_matrix = get_3d_points(disparity_map, disparity_map_matrix)
    remove_outliers(location_map)

    object_points = apply_mask(location_map, object_masks)

    poses = []
    images = []

    for item in object_points:
        centre, axis_point_1, axis_point_2 = get_axis(item)
        rotation = get_object_rotation(axis_point_1, axis_point_2)

        poses.append({'location': centre, 'rotation': rotation})
        colour = (randint(0, 255), randint(0, 255), randint(0, 255))

        image = img_1.copy()

        draw_axis(image, item, centre, axis_point_1, lambda_map, disparity_map_matrix, colour=colour)

        images.append(image)

    return poses, images
