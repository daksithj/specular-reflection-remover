from oct2py import octave

OCTAVE_PATH='./MultiImage/Octave'

def multi_image_removal(image_1, image_2, image_3):
    octave.addpath(OCTAVE_PATH)
    image_1 = image_1 / 255
    image_2 = image_2 / 255
    image_3 = image_3 / 255
    d_f1 = octave.MultiImageSpectralDifference(image_1, image_2, image_3)
    d_f2 = octave.MultiImageSpectralDifference(image_2, image_1, image_3)
    d_f3 = octave.MultiImageSpectralDifference(image_3, image_2, image_1)

    return (d_f1), (image_1 - d_f1), (d_f2), (image_2 - d_f2), (d_f3), (image_3 - d_f3)






