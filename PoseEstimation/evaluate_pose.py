import numpy as np


def get_closest_location(real_locations, location):

    min_dist = 0
    closest_loc = None
    closest_index = 0

    for i, real_loc in enumerate(real_locations):

        squared_dist = np.sum((real_loc - location) ** 2, axis=0)
        dist = np.sqrt(squared_dist)

        if closest_loc is None or dist < min_dist:
            min_dist = dist
            closest_loc = real_loc
            closest_index = i

    return closest_loc, closest_index


def calculate_error(real_locations, locations):

    total_error = 0

    count = 0

    fail_rate = 0

    for loc in real_locations:
        closest, index = get_closest_location(locations, loc)
        if closest is not None:
            squared_dist = np.sum((closest - loc) ** 2, axis=0)
            dist = np.sqrt(squared_dist)
            if dist > 4:
                fail_rate +=1
            total_error += dist
            locations.pop(index)
            count += 1
        else:
            fail_rate += 1

    return total_error/count, fail_rate
