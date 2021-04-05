import os
import glob
from tensorflow import keras
import numpy as np

from DeepLearning.tools import convert_image_to_hsv, prepare_input, translate_image


def get_diffuse(image_1, image_2, model_name=None):

    if model_name is None:
        file_name = max(glob.iglob('Models/Generator/*.h5'), key=os.path.getctime)
    else:
        file_name = 'Models/Generator/' + model_name + '.h5'

    generator = keras.models.load_model(file_name)

    hsv_image_1 = convert_image_to_hsv(image_1, specular=True, gray=False)
    input_image_1 = prepare_input(hsv_image_1)

    hsv_image_2 = convert_image_to_hsv(image_2, specular=True, gray=False)
    input_image_2 = prepare_input(hsv_image_2)

    input_image = np.concatenate([input_image_1, input_image_2], axis=2)

    translated_image = generator.predict(input_image)
    translated_image = translate_image(translated_image)

    mid_point = int (translated_image.shape[1]/2)

    output_1 = np.uint8(translated_image[:, :mid_point, :])
    output_2 = np.uint8(translated_image[:, mid_point:, :])

    return output_1, output_2


