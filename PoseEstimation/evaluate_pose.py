import numpy as np
import pickle as pkl


def read_actual_locations(folder, data_num):

    loc_file = open(f'{folder}/location_{data_num}', 'rb')
    locations = pkl.load(loc_file)
    loc_file.close()

    return np.asarray(locations)


def get_closest_location(real_locations, location):

    if real_locations is None:
        real_locations = read_actual_locations('Dataset/Test', 0)

    min_dist = 0
    closest_loc = None

    for real_loc in real_locations:

        squared_dist = np.sum((real_loc - location) ** 2, axis=0)
        dist = np.sqrt(squared_dist)

        if closest_loc is None or dist < min_dist:
            min_dist = dist
            closest_loc = real_loc

    return closest_loc


def calculate_error(real_locations, locations):

    total_error = 0

    for loc in locations:
        closest = get_closest_location(real_locations, loc)
        squared_dist = np.sum((closest - loc) ** 2, axis=0)
        dist = np.sqrt(squared_dist)
        total_error += dist

    return total_error/len(locations)
