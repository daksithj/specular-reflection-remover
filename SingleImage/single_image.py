from oct2py import octave
import numpy as np
import cv2

OCTAVE_PATH = './SingleImage/Octave'


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def single_image_removal(image):
    octave.addpath(OCTAVE_PATH)
    image = image / 255
    diffuse, specular = octave.SpecularRemover(image, nout=2)
    diffuse = cv2.normalize(diffuse, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    diffuse = apply_brightness_contrast(diffuse, 0, 30)

    gray = cv2.cvtColor(diffuse, cv2.COLOR_BGR2GRAY)

    _, blackAndWhite = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, None, None, None, 4, cv2.CV_32S)
    sizes = stats[1:, -1]
    mask = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 20:
            mask[labels == i + 1] = 255

    mask = cv2.bitwise_not(mask)

    mask = np.dstack([mask, mask, mask]) / 255

    diffuse = diffuse * mask
    diffuse = np.asarray(diffuse).astype(np.uint8)

    return (diffuse), (specular)






