import cv2
import numpy as np


# Segment objects and get masks for each individual objects
def get_object_masks(img, k_1=7, k_2=2):

    # Convert to HSV format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    flat_image = img.reshape((-1, 3))

    flat_image = np.float32(flat_image)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Get initial clusters
    _, label, center = cv2.kmeans(flat_image, k_1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    label = label.flatten()
    res = center[label]
    res2 = res.reshape(img.shape)

    # Convert back to BGR
    img = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)

    flat_image = img.reshape((-1, 3))

    flat_image = np.float32(flat_image)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Get K means again to reduce further
    _, label, center = cv2.kmeans(flat_image, k_2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    label = label.flatten()

    colour_masks = []
    background_idx = 0
    background_weight = 0

    # Based on the colour create masks for each colour
    for col in range(k_2):
        mask = np.zeros(shape=label.shape)
        point_count = 0
        for idx, point in enumerate(label):
            if point == col:
                mask[idx] = 255
                point_count += 1
        # Identify the background
        if point_count > background_weight:
            background_idx = col
            background_weight = point_count
        colour_masks.append(mask)

    colour_masks.pop(background_idx)

    object_masks = []

    for colour_comp in colour_masks:
        colour_comp = colour_comp.reshape((img.shape[0], img.shape[1]))
        colour_comp = np.uint8(colour_comp)
        # Separate each colour masks to objects by identifying connected components
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(colour_comp, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        min_size = 400
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                mask = np.zeros(output.shape)
                mask[output == i + 1] = 1
                object_masks.append(mask)

    return object_masks
