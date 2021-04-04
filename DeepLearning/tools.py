import numpy as np
import cv2


def convert_image_to_hsv(image, gray=True, specular=True):
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


def receptive_field_calc(num_layers):
    def receptive(output_size, kernel_size, stride_size):
        return (output_size - 1) * stride_size + kernel_size

    rf = receptive(1, 4, 1)

    rf = receptive(rf, 4, 1)

    for i in range(num_layers):
        rf = receptive(rf, 4, 2)
    return rf


def translate_image(image):

    image = 0.5 * image + 0.5
    image = image * 255
    image = np.asarray(image).astype(np.uint8)
    image = np.squeeze(image, axis=0)

    return image


def prepare_input(image):
    image = image / 127.5 - 1.0
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    return image
