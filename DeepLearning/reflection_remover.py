import os
import glob
from tensorflow import keras
import numpy as np

from DeepLearning.tools import convert_image_to_hsv, prepare_input, translate_image


def convert_row(input_image_1, input_image_2, output_1, output_2, generator, input_x_1, input_x_2, slice_1, slice_2,
                output_x_1, output_x_2):

    div = output_1.shape[0] // 3
    gap = 30

    input_image = np.concatenate([input_image_1[:, input_x_1: input_x_2, :256, :],
                                  input_image_2[:, input_x_1: input_x_2, :256, :]], axis=2)

    translated_image = generator.predict(input_image)
    translated_image = translate_image(translated_image)

    output_1[output_x_1:output_x_2, :256, :] = translated_image[slice_1:slice_2, :256, :]
    output_2[output_x_1:output_x_2, :256, :] = translated_image[slice_1:slice_2, 256:512, :]

    input_image = np.concatenate([input_image_1[:, input_x_1: input_x_2, -256:, :],
                                  input_image_2[:, input_x_1: input_x_2, -256:, :]], axis=2)

    translated_image = generator.predict(input_image)
    translated_image = translate_image(translated_image)

    output_1[output_x_1:output_x_2, -256:, :] = translated_image[slice_1:slice_2, :256, :]
    output_2[output_x_1:output_x_2, -256:, :] = translated_image[slice_1:slice_2, 256:512, :]

    input_image = np.concatenate([input_image_1[:, input_x_1: input_x_2, div:div + 256, :],
                                  input_image_2[:, input_x_1: input_x_2, div:div + 256, :]], axis=2)

    translated_image = generator.predict(input_image)
    translated_image = translate_image(translated_image)

    output_slice_1 = translated_image[slice_1:slice_2, gap:256 - gap, :]
    output_slice_2 = translated_image[slice_1:slice_2, 256 + gap:512 - gap, :]

    output_1[output_x_1:output_x_2, div + gap:div + 256 - gap, :] = output_slice_1
    output_2[output_x_1:output_x_2, div + gap:div + 256 - gap, :] = output_slice_2


def get_diffuse(image_1, image_2, model_name=None):

    if model_name is None:
        file_name = max(glob.iglob('Models/Generator/*.h5'), key=os.path.getctime)
    else:
        file_name = 'Models/Generator/' + model_name + '.h5'

    generator = keras.models.load_model(file_name)

    div = image_1.shape[0] // 3
    gap = 30

    hsv_image_1 = convert_image_to_hsv(image_1, specular=True, gray=False)
    input_image_1 = prepare_input(hsv_image_1)
    output_1 = np.zeros(shape=hsv_image_1.shape)

    hsv_image_2 = convert_image_to_hsv(image_2, specular=True, gray=False)
    input_image_2 = prepare_input(hsv_image_2)
    output_2 = np.zeros(shape=hsv_image_2.shape)

    convert_row(input_image_1, input_image_2, output_1, output_2, generator, 0, 256, 0, 256, 0, 256)
    convert_row(input_image_1, input_image_2, output_1, output_2, generator, 256, 512, 0, 256, 256, 512)
    convert_row(input_image_1, input_image_2, output_1, output_2, generator, div, div + 256, gap, 256 - gap, div + gap,
                div + 256 - gap)

    output_1 = np.uint8(output_1)
    output_2 = np.uint8(output_2)

    return output_1, output_2


