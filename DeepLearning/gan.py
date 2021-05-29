import tensorflow as tf
from tensorflow import keras
from DeepLearning.dataset_gan import ImageDataSet
from DeepLearning.tools import translate_image
from PoseEstimation.pose_estimator import get_disparity_map_matrix, get_pose
from PoseEstimation.evaluate_pose import read_actual_locations, calculate_error
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as keras_backend
import numpy as np
import random
import os
import glob
import cv2
import matplotlib.pyplot as plt

batch_size = 1
channels = 3
image_pairs = 2
image_size = 512
feature_patch_size = 32
image_shape = (image_size, image_size*image_pairs, channels)
patch_shape = (feature_patch_size, feature_patch_size, 3)
filter_thresh = 512
summary_location = 'Summary/'
model_location = 'Models/'
matrix_location = 'Dataset/Matrix'
g_lr = 0.0002
d_lr = 0.0002

g_bias = False
d_bias = False

discriminator_loss = 'binary_crossentropy'
generator_loss = 'mae'
use_vgg_loss = True
use_location_loss = False

discriminator_loss_weight = 1
generator_loss_weight = 80
vgg_loss_weight = 20
location_loss_weight = 20

disparity_map_matrix = get_disparity_map_matrix(matrix_location, (image_size, image_size))

VGG_net = tf.keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=patch_shape,
        pooling=None,
        classifier_activation="softmax",
    )


def gen_encoder_block(input_layer, filters, batch_norm=True):
    k_init = keras.initializers.RandomNormal(stddev=0.02)

    layer = layers.Conv2D(filters=filters, kernel_size=4, strides=2, use_bias=g_bias, padding='same',
                          kernel_initializer=k_init)(input_layer)

    if batch_norm:
        layer = layers.BatchNormalization()(layer, training=True)

    layer = layers.LeakyReLU(alpha=0.2)(layer)

    return layer


def gen_decoder_block(input_layer, skip_layer, filters, dropout=True):

    k_init = keras.initializers.RandomNormal(stddev=0.02)

    layer = layers.Conv2DTranspose(filters, kernel_size=4, strides=2, use_bias=g_bias, padding='same',
                                   kernel_initializer=k_init)(input_layer)

    layer = layers.BatchNormalization()(layer, training=True)

    if dropout:
        layer = layers.Dropout(0.5)(layer, training=True)

    layer = layers.Concatenate()([layer, skip_layer])

    layer = layers.Activation('relu')(layer)

    return layer


def build_generator():
    k_init = keras.initializers.RandomNormal(stddev=0.02)

    input_layer = layers.Input(shape=image_shape)

    layer_num = int(np.log2(image_size)) - 1

    filter_size = image_size // 16

    # Encoder layers

    encoding_layers = []

    down = input_layer

    batch_norm = False

    for _ in range(layer_num):

        down = gen_encoder_block(down, filter_size, batch_norm=batch_norm)

        if not batch_norm:
            batch_norm = True

        if filter_size < filter_thresh:
            filter_size = filter_size * 2

        encoding_layers.append(down)

    # Inner layer
    inner = layers.Conv2D(filters=filter_thresh, kernel_size=4, strides=2, use_bias=g_bias, padding='same',
                          kernel_initializer=k_init)(down)

    up = layers.Activation('relu')(inner)

    # Decoder layers

    encoding_layers.reverse()

    drop_count = layer_num - int(np.log2(filter_size))

    dropout = True

    for down_layer in encoding_layers:

        up = gen_decoder_block(up, down_layer, down_layer.shape[3], dropout=dropout)

        if drop_count == 0:
            dropout = False
        else:
            drop_count -= 1

    # Output layer
    outer = layers.Conv2DTranspose(channels, kernel_size=4, strides=2, use_bias=g_bias, padding='same',
                                   kernel_initializer=k_init)(up)
    output = layers.Activation(activation='tanh')(outer)

    model = Model(input_layer, output)

    return model


def discriminator_block(input_layer, filters, strides, k_init, batch_norm=True):

    layer = layers.Conv2D(filters=filters, kernel_size=4, strides=strides, use_bias=d_bias, padding='same',
                          kernel_initializer=k_init)(input_layer)

    if batch_norm:
        layer = layers.BatchNormalization()(layer)

    layer = layers.LeakyReLU(alpha=0.2)(layer)

    return layer


def build_discriminator():

    k_init = keras.initializers.RandomNormal(stddev=0.02)

    layer_num = int(np.log2(image_size)) - 3

    filter_size = image_size // 16

    input_source = layers.Input(shape=image_shape)

    input_target = layers.Input(shape=image_shape)

    input_layer = layers.Concatenate()([input_source, input_target])

    # Down layers
    batch_norm = False

    strides = 2

    down = input_layer

    for num in range(layer_num):

        if num == layer_num - 1:
            strides = 1

        down = discriminator_block(down, filter_size, strides=strides, k_init=k_init, batch_norm=batch_norm)

        if not batch_norm:
            batch_norm = True

        if filter_size < filter_thresh:
            filter_size = filter_size * 2

    output = layers.Conv2D(filters=1, kernel_size=4, use_bias=d_bias, padding='same',
                           kernel_initializer=k_init)(down)

    output = layers.Activation(activation='sigmoid')(output)

    model = Model([input_source, input_target], output)

    optimizer = keras.optimizers.Adam(lr=d_lr, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, loss_weights=[0.5])

    return model


def match_features(image_1, image_2, patch_size):

    image_1 = translate_image(image_1)
    image_2 = translate_image(image_2)

    orb = cv2.ORB_create(nfeatures=1000, edgeThreshold=1, nlevels=8, fastThreshold=20, scoreType=cv2.ORB_FAST_SCORE)

    key_points_1, desc_1 = orb.detectAndCompute(image_1, None)
    key_points_2, desc_2 = orb.detectAndCompute(image_2, None)

    desc_1 = np.float32(desc_1)
    desc_2 = np.float32(desc_2)

    index_params = dict(algorithm=1, trees=5)

    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_1, desc_2, k=2)

    match_points = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            match_points.append(m)

    random.shuffle(match_points)

    for match in match_points:

        image_1_point = match.queryIdx
        image_2_point = match.trainIdx

        point1_x, point1_y = key_points_1[image_1_point].pt
        point2_x, point2_y = key_points_2[image_2_point].pt

        point1_x = int(point1_x)
        point2_x = int(point2_x)
        point1_y = int(point1_y)
        point2_y = int(point2_y)

        patch_half = patch_size // 2
        shape = image_1.shape[0]

        image_1_x1 = int(point1_x - patch_half)
        image_1_x2 = int(point1_x + patch_half)
        image_1_y1 = int(point1_y - patch_half)
        image_1_y2 = int(point1_y + patch_half)

        image_2_x1 = int(point2_x - patch_half)
        image_2_x2 = int(point2_x + patch_half)
        image_2_y1 = int(point2_y - patch_half)
        image_2_y2 = int(point2_y + patch_half)

        if image_1_x1 < 0 or image_1_y1 < 0 or image_2_x1 < 0 or image_2_y1 < 0:
            continue

        if image_1_x2 > shape or image_1_y2 > shape or image_2_x2 > shape or image_2_y2 > shape:
            continue

        return (image_1_x1, image_1_x2, image_1_y1, image_1_y2), (image_2_x1, image_2_x2, image_2_y1, image_2_y2)

    return None


def build_patch_VGG(y_true, y_predict):

    total_loss = 0

    target_1 = y_true[:, :, 0:image_size, :]
    target_2 = y_true[:, :, image_size: 2 * image_size, :]

    predict_1 = y_predict[:, :, 0:image_size, :]
    predict_2 = y_predict[:, :, image_size: 2 * image_size, :]

    coordinates = match_features(keras_backend.eval(target_1), keras_backend.eval(target_2), feature_patch_size)

    if coordinates is None:
        return keras_backend.mean(keras.metrics.mean_absolute_error(y_true, y_predict))

    (x1, x2, y1, y2), (x3, x4, y3, y4) = coordinates

    real_1 = target_1[:, x1:x2, y1:y2, :]
    fake_1 = predict_1[:, x1:x2, y1:y2, :]

    real_2 = target_2[:, x3:x4, y3:y4, :]
    fake_2 = predict_2[:, x3:x4, y3:y4, :]

    for layer in VGG_net.layers:
        real_1 = layer(real_1)
        fake_1 = layer(fake_1)
        total_loss += keras_backend.mean(keras.metrics.mean_absolute_error(real_1, fake_1))

        real_2 = layer(real_2)
        fake_2 = layer(fake_2)
        total_loss += keras_backend.mean(keras.metrics.mean_absolute_error(real_2, fake_2))

    return total_loss/2


def location_loss(locations, y_predict):

    locations = keras_backend.eval(locations)

    predict_1 = y_predict[:, :, 0:image_size, :]
    predict_2 = y_predict[:, :, image_size: 2 * image_size, :]

    predict_1 = keras_backend.eval(predict_1)
    predict_2 = keras_backend.eval(predict_2)

    predict_1 = np.squeeze(predict_1, axis=0)
    predict_2 = np.squeeze(predict_2, axis=0)

    predict_1 = np.uint8(predict_1)
    predict_2 = np.uint8(predict_2)

    poses = get_pose(predict_1, predict_2, disparity_map_matrix)

    num_obj_error = np.abs((len(poses) - len(locations)))

    if len(poses) > 0:
        location_set = []

        for pose in poses:
            location_set.append(pose['location'])

        location_error = calculate_error(np.asarray(locations), location_set)

        total_error = num_obj_error * location_error
    else:
        total_error = 0

    return total_error


def build_gan(gen_model, dis_model):

    for layer in dis_model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    input_layer = layers.Input(shape=image_shape)

    gen_out = gen_model(input_layer)

    dis_out = dis_model([input_layer, gen_out])

    model_output = [dis_out, gen_out]
    loss = [discriminator_loss, generator_loss]
    loss_weights = [discriminator_loss_weight, generator_loss_weight]

    if use_vgg_loss:
        loss.append(build_patch_VGG)
        loss_weights.append(vgg_loss_weight)
        model_output.append(gen_out)

    if use_location_loss:
        loss.append(location_loss)
        loss_weights.append(location_loss_weight)
        model_output.append(gen_out)

    model = Model(input_layer, model_output)

    optimizer = keras.optimizers.Adam(lr=g_lr, beta_1=0.5)

    model.compile(loss=loss, optimizer=optimizer, loss_weights=loss_weights, run_eagerly=True)

    return model


def generate_output(gen_model, model_input, patch_size):

    output = gen_model.predict(model_input)
    labels = tf.zeros((len(output), patch_size, patch_size*image_pairs, 1))

    return output, labels


def end_of_epoch(gen_model, data_set, patch_size):

    data_len = int(data_set.__len__())

    specular, diffuse = data_set.__getitem__(random.randint(0, data_len))

    generate_output(gen_model, specular, patch_size)


def generate_test_sample(samples=1, pair=True):

    image_data_set = ImageDataSet(batch_size=samples)

    specular_image, target = image_data_set.__getitem__(0)

    diffuse_image, _ = target

    specular_image = np.asarray(specular_image)

    if pair:
        diffuse_image = np.asarray(diffuse_image)
        return specular_image, diffuse_image
    else:
        return specular_image


def training_summary(gen_model, summary_folder, epoch, step=0, samples=1):

    specular_real, diffuse_real = generate_test_sample(samples, pair=True)

    gen_out, _ = generate_output(gen_model, specular_real, 1)

    specular_real = (specular_real + 1) / 2.0
    diffuse_real = (diffuse_real + 1) / 2.0
    gen_out = (gen_out + 1) / 2.0

    for i in range(samples):
        plt.subplot(3, samples, 1 + i)
        plt.axis('off')
        plt.imshow(specular_real[i])

    for i in range(samples):
        plt.subplot(3, samples, 1 + samples + i)
        plt.axis('off')
        plt.imshow(gen_out[i])

    for i in range(samples):
        plt.subplot(3, samples, 1 + samples * 2 + i)
        plt.axis('off')
        plt.imshow(diffuse_real[i])

    filename = '/plot_%04d_%05d.png' % ((epoch + 1), step)
    filename = summary_folder + filename
    plt.savefig(filename)
    plt.close()
    print('>Saved summary to: %s ' % filename)


def train_gan(gen_model, dis_model, gan_model, data_set, epochs=100, cont_train=False):

    try:
        file_name = max(glob.iglob('Models/Generator/*.h5'), key=os.path.getctime)
        file_name = os.path.basename(file_name)
    except ValueError:
        file_name = '0.h5'

    if cont_train:
        gen_model = keras.models.load_model('Models/Generator/' + file_name)
        dis_model = keras.models.load_model('Models/Discriminator/' + file_name)
    else:
        file_name = os.path.splitext(file_name)[0]
        file_name = str(int(file_name) + 1)
        file_name = file_name + '.h5'

    summary_folder = 'Summary/' + os.path.splitext(file_name)[0]

    if not os.path.exists(summary_folder):
        os.makedirs(summary_folder)

    data_len = int(data_set.__len__())

    patch_size = dis_model.output_shape[1]

    for epoch in range(epochs):
        for idx in range(data_len):

            specular_real, target = data_set.__getitem__(idx)

            diffuse_real, locations = target

            labels_real = tf.ones((batch_size, patch_size, patch_size*image_pairs, 1))

            gen_out, gen_label = generate_output(gen_model, specular_real, patch_size)

            dis_loss_real = dis_model.train_on_batch([specular_real, diffuse_real], labels_real)

            dis_loss_gen = dis_model.train_on_batch([specular_real, gen_out], gen_label)

            output_target = [labels_real, diffuse_real]

            if use_vgg_loss:
                output_target.append(diffuse_real)

            if use_location_loss:
                output_target.append(locations)

            gan_losses = gan_model.train_on_batch(specular_real, output_target)

            gan_loss = gan_losses[0]

            print('Epoch: %d  Step: %d Discriminator loss (target): [%.3f] Discriminator loss (Specular): [%.3f] '
                  'Generator loss: [%.3f]' % (epoch+1, idx + 1, dis_loss_real, dis_loss_gen, gan_loss))

            if idx % 100 == 0:
                training_summary(gen_model, summary_folder, epoch, idx, batch_size)
                gen_model.save('Models/Generator/' + file_name)
                dis_model.save('Models/Discriminator/' + file_name)

        data_set.on_epoch_end()

        training_summary(gen_model, summary_folder, epoch, batch_size)
        gen_model.save('Models/Generator/' + file_name)
        dis_model.save('Models/Discriminator/' + file_name)


def start_training(cont=False):
    generator = build_generator()

    discriminator = build_discriminator()

    gan = build_gan(generator, discriminator)

    data_set = ImageDataSet(batch_size)

    train_gan(generator, discriminator, gan, data_set, cont_train=cont)


def test_generator(model_name=None):

    if model_name is None:
        file_name = max(glob.iglob('Models/Generator/*.h5'), key=os.path.getctime)
    else:
        file_name = 'Models/Generator/' + model_name + '.h5'

    generator = keras.models.load_model(file_name)

    sample = generate_test_sample(1, pair=False)

    output = generator.predict(sample)

    output = translate_image(output)

    cv2.imshow("Output", output)
    cv2.waitKey(0) & 0xFF


start_training(True)
