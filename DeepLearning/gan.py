import tensorflow as tf
from tensorflow import keras
from DeepLearning.dataset_gan import ImageDataSet
from PoseEstimation.pose_estimator import get_disparity_map_matrix, get_pose
from PoseEstimation.evaluate_pose import calculate_error
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as keras_backend
import numpy as np
import random
import json
import os
import cv2
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

network_directory = 'DeepLearning/Networks/'


class SpecToPoseNet:

    def __init__(self, network_name, image_dataset=None):

        self.batch_size = 1

        self.network_name = network_name
        self.network_location = network_directory + self.network_name + '/'
        self.summary_location = self.network_location + 'Summary/'
        self.network_data_location = self.network_location + 'network_data.json'

        if isinstance(image_dataset, ImageDataSet):
            self.image_dataset = image_dataset
        else:
            self.image_dataset = self.set_image_dataset()

        self.channels = self.image_dataset.channels
        self.image_pairs = self.image_dataset.pairs
        self.image_size = self.image_dataset.image_size

        self.image_shape = (self.image_size, self.image_size * self.image_pairs, self.channels)

        self.matrix_location = self.image_dataset.matrix_dir

        self.feature_patch_size = 32
        self.patch_shape = (self.feature_patch_size, self.feature_patch_size, 3)

        self.filter_thresh = 512
        self.g_lr = 0.0002
        self.d_lr = 0.0002

        self.g_bias = False
        self.d_bias = False

        self.discriminator_loss = 'binary_crossentropy'
        self.generator_loss = 'mae'
        self.use_vgg_loss = False
        self.use_location_loss = False

        self.discriminator_loss_weight = 1
        self.generator_loss_weight = 80
        self.vgg_loss_weight = 20
        self.location_loss_weight = 20

        self.disparity_map_matrix = get_disparity_map_matrix(self.matrix_location, (self.image_size, self.image_size))

        self.VGG_net = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=self.patch_shape,
            pooling=None,
            classifier_activation="softmax",
            )

        self.loaded, self.generator, self.discriminator, self.gan_model = self.load_models()
        self.network_data = self.load_network_data()

    def load_models(self):
        try:
            generator = keras.models.load_model(self.network_location + "generator.h5")
            discriminator = keras.models.load_model(self.network_location + "discriminator.h5")

            gen_shape = generator.layers[0].input_shape[0][1:]
            dis_shape = discriminator.layers[0].input_shape[0][1:]

            if not (gen_shape == self.image_shape and dis_shape == self.image_shape):
                raise ValueError("The loaded dataset and loaded model dimensions do not match")

            gan_model = self.build_gan(generator, discriminator)

            return True, generator, discriminator, gan_model

        except (ImportError, IOError):
            generator = self.build_generator()
            discriminator = self.build_discriminator()

            gan_model = self.build_gan(generator, discriminator)

            if not os.path.exists(network_directory):
                os.mkdir(network_directory)

            if not os.path.exists(self.network_location):
                os.mkdir(self.network_location)

            generator.save(self.network_location + 'generator.h5')
            discriminator.save(self.network_location + 'discriminator.h5')

            return False, generator, discriminator, gan_model

    def load_network_data(self):

        if os.path.exists(self.network_data_location):
            with open(self.network_data_location) as f:
                network_data = json.load(f)
        else:
            network_data = {
                'name': self.network_name,
                'image_size': self.image_size,
                'channels': self.channels,
                'pairs': int(self.image_pairs),
                'epochs': 0,
                'steps': 0,
                'dataset_name': self.image_dataset.dataset_name
            }

        with open(self.network_data_location, 'w') as f:
            json.dump(network_data, f)
        return network_data

    def set_image_dataset(self):
        if os.path.exists(self.network_data_location):
            with open(self.network_data_location) as f:
                network_data = json.load(f)
                return ImageDataSet(network_data['dataset_name'])
        else:
            raise ValueError("Cannot find dataset for model.")

    def reset_network(self):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.gan_model = self.build_gan(self.generator, self.discriminator)

        self.network_data = {
            'name': self.network_name,
            'image_size': self.image_size,
            'channels': self.channels,
            'pairs': int(self.image_pairs),
            'epochs': 0,
            'steps': 0,
            "dataset_name": self.image_dataset.dataset_name
        }

        if os.path.exists(self.summary_location):
            os.rmdir(self.summary_location)

    def gen_encoder_block(self, input_layer, filters, batch_norm=True):
        k_init = keras.initializers.RandomNormal(stddev=0.02)

        layer = layers.Conv2D(filters=filters, kernel_size=4, strides=2, use_bias=self.g_bias, padding='same',
                              kernel_initializer=k_init)(input_layer)

        if batch_norm:
            layer = layers.BatchNormalization()(layer, training=True)

        layer = layers.LeakyReLU(alpha=0.2)(layer)

        return layer

    def gen_decoder_block(self, input_layer, skip_layer, filters, dropout=True):

        k_init = keras.initializers.RandomNormal(stddev=0.02)

        layer = layers.Conv2DTranspose(filters, kernel_size=4, strides=2, use_bias=self.g_bias, padding='same',
                                       kernel_initializer=k_init)(input_layer)

        layer = layers.BatchNormalization()(layer, training=True)

        if dropout:
            layer = layers.Dropout(0.5)(layer, training=True)

        layer = layers.Concatenate()([layer, skip_layer])

        layer = layers.Activation('relu')(layer)

        return layer

    def build_generator(self):
        k_init = keras.initializers.RandomNormal(stddev=0.02)

        input_layer = layers.Input(shape=self.image_shape)

        layer_num = int(np.log2(self.image_size)) - 1

        filter_size = self.image_size // 16

        # Encoder layers

        encoding_layers = []

        down = input_layer

        batch_norm = False

        for _ in range(layer_num):

            down = self.gen_encoder_block(down, filter_size, batch_norm=batch_norm)

            if not batch_norm:
                batch_norm = True

            if filter_size < self.filter_thresh:
                filter_size = filter_size * 2

            encoding_layers.append(down)

        # Inner layer
        inner = layers.Conv2D(filters=self.filter_thresh, kernel_size=4, strides=2, use_bias=self.g_bias,
                              padding='same', kernel_initializer=k_init)(down)
        up = layers.Activation('relu')(inner)

        # Decoder layers

        encoding_layers.reverse()

        drop_count = layer_num - int(np.log2(filter_size))

        dropout = True

        for down_layer in encoding_layers:

            up = self.gen_decoder_block(up, down_layer, down_layer.shape[3], dropout=dropout)

            if drop_count == 0:
                dropout = False
            else:
                drop_count -= 1

        # Output layer
        outer = layers.Conv2DTranspose(self.channels, kernel_size=4, strides=2, use_bias=self.g_bias, padding='same',
                                       kernel_initializer=k_init)(up)
        output = layers.Activation(activation='tanh')(outer)

        model = Model(input_layer, output)

        return model

    def discriminator_block(self, input_layer, filters, strides, k_init, batch_norm=True):

        layer = layers.Conv2D(filters=filters, kernel_size=4, strides=strides, use_bias=self.d_bias, padding='same',
                              kernel_initializer=k_init)(input_layer)

        if batch_norm:
            layer = layers.BatchNormalization()(layer)

        layer = layers.LeakyReLU(alpha=0.2)(layer)

        return layer

    def build_discriminator(self):

        k_init = keras.initializers.RandomNormal(stddev=0.02)

        layer_num = int(np.log2(self.image_size)) - 3

        filter_size = self.image_size // 16

        input_source = layers.Input(shape=self.image_shape)

        input_target = layers.Input(shape=self.image_shape)

        input_layer = layers.Concatenate()([input_source, input_target])

        # Down layers
        batch_norm = False

        strides = 2

        down = input_layer

        for num in range(layer_num):

            if num == layer_num - 1:
                strides = 1

            down = self.discriminator_block(down, filter_size, strides=strides, k_init=k_init, batch_norm=batch_norm)

            if not batch_norm:
                batch_norm = True

            if filter_size < self.filter_thresh:
                filter_size = filter_size * 2

        output = layers.Conv2D(filters=1, kernel_size=4, use_bias=self.d_bias, padding='same',
                               kernel_initializer=k_init)(down)

        output = layers.Activation(activation='sigmoid')(output)

        model = Model([input_source, input_target], output)

        optimizer = keras.optimizers.Adam(lr=self.d_lr, beta_1=0.5)

        model.compile(loss='binary_crossentropy', optimizer=optimizer, loss_weights=[0.5])

        return model

    @staticmethod
    def match_features(image_1, image_2, patch_size):

        image_1 = ImageDataSet.translate_image(image_1)
        image_2 = ImageDataSet.translate_image(image_2)

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

    def build_patch_VGG(self, y_true, y_predict):

        total_loss = 0

        target_1 = y_true[:, :, 0:self.image_size, :]
        target_2 = y_true[:, :, self.image_size: 2 * self.image_size, :]

        predict_1 = y_predict[:, :, 0:self.image_size, :]
        predict_2 = y_predict[:, :, self.image_size: 2 * self.image_size, :]

        coordinates = self.match_features(keras_backend.eval(target_1), keras_backend.eval(target_2),
                                          self.feature_patch_size)

        if coordinates is None:
            return keras_backend.mean(keras.metrics.mean_absolute_error(y_true, y_predict))

        (x1, x2, y1, y2), (x3, x4, y3, y4) = coordinates

        real_1 = target_1[:, x1:x2, y1:y2, :]
        fake_1 = predict_1[:, x1:x2, y1:y2, :]

        real_2 = target_2[:, x3:x4, y3:y4, :]
        fake_2 = predict_2[:, x3:x4, y3:y4, :]

        for layer in self.VGG_net.layers:
            real_1 = layer(real_1)
            fake_1 = layer(fake_1)
            total_loss += keras_backend.mean(keras.metrics.mean_absolute_error(real_1, fake_1))

            real_2 = layer(real_2)
            fake_2 = layer(fake_2)
            total_loss += keras_backend.mean(keras.metrics.mean_absolute_error(real_2, fake_2))

        return total_loss/2

    def location_loss(self, locations, y_predict):

        locations = keras_backend.eval(locations)

        predict_1 = y_predict[:, :, 0:self.image_size, :]
        predict_2 = y_predict[:, :, self.image_size: 2 * self.image_size, :]

        predict_1 = keras_backend.eval(predict_1)
        predict_2 = keras_backend.eval(predict_2)

        predict_1 = np.squeeze(predict_1, axis=0)
        predict_2 = np.squeeze(predict_2, axis=0)

        predict_1 = np.uint8(predict_1)
        predict_2 = np.uint8(predict_2)

        poses = get_pose(predict_1, predict_2, self.disparity_map_matrix)

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

    def build_gan(self, gen_model, dis_model):

        for layer in dis_model.layers:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        input_layer = layers.Input(shape=self.image_shape)

        gen_out = gen_model(input_layer)

        dis_out = dis_model([input_layer, gen_out])

        model_output = [dis_out, gen_out]
        loss = [self.discriminator_loss, self.generator_loss]
        loss_weights = [self.discriminator_loss_weight, self.generator_loss_weight]

        if self.use_vgg_loss:
            loss.append(self.build_patch_VGG)
            loss_weights.append(self.vgg_loss_weight)
            model_output.append(gen_out)

        if self.use_location_loss:
            loss.append(self.location_loss)
            loss_weights.append(self.location_loss_weight)
            model_output.append(gen_out)

        model = Model(input_layer, model_output)

        optimizer = keras.optimizers.Adam(lr=self.g_lr, beta_1=0.5)

        model.compile(loss=loss, optimizer=optimizer, loss_weights=loss_weights, run_eagerly=True)

        return model

    def generate_output(self, gen_model, model_input, patch_size):

        output = gen_model.predict(model_input)
        labels = tf.zeros((len(output), patch_size, patch_size*self.image_pairs, 1))

        return output, labels

    def end_of_epoch(self, gen_model, data_set, patch_size):

        data_len = int(data_set.__len__())

        specular, diffuse = data_set.__getitem__(random.randint(0, data_len))

        self.generate_output(gen_model, specular, patch_size)

    def generate_test_sample(self,  pair=True):

        specular_image, target = self.image_dataset.__getitem__(0)

        diffuse_image, _ = target

        specular_image = np.asarray(specular_image)

        if pair:
            diffuse_image = np.asarray(diffuse_image)
            return specular_image, diffuse_image
        else:
            return specular_image

    def training_summary(self, gen_model, summary_folder, epoch, step=0, samples=1):

        specular_real, diffuse_real = self.generate_test_sample(pair=True)

        gen_out, _ = self.generate_output(gen_model, specular_real, 1)

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

    def train_gan(self, toggle_vgg=-1, toggle_location_loss=-1, epochs=100, gui=None):

        if not os.path.exists(self.summary_location):
            os.makedirs(self.summary_location)

        data_len = int(self.image_dataset.__len__())

        patch_size = self.discriminator.output_shape[1]

        total_epochs = int(self.network_data['epochs'])

        steps = int(self.network_data['steps'])

        total_steps = int(epochs * data_len)

        epoch = total_epochs

        while epoch < (total_epochs + epochs):

            if epoch - total_epochs == toggle_vgg:
                self.use_vgg_loss = not self.use_vgg_loss
                self.gan_model = self.build_gan(self.generator, self.discriminator)

            if epoch - total_epochs == toggle_location_loss:
                self.use_location_loss = not self.use_location_loss
                self.gan_model = self.build_gan(self.generator, self.discriminator)

            while steps < data_len:

                specular_real, target = self.image_dataset.__getitem__(steps)

                diffuse_real, locations = target

                labels_real = tf.ones((self.batch_size, patch_size, patch_size*self.image_pairs, 1))

                gen_out, gen_label = self.generate_output(self.generator, specular_real, patch_size)

                dis_loss_real = self.discriminator.train_on_batch([specular_real, diffuse_real], labels_real)

                dis_loss_gen = self.discriminator.train_on_batch([specular_real, gen_out], gen_label)

                output_target = [labels_real, diffuse_real]

                if self.use_vgg_loss:
                    output_target.append(diffuse_real)

                if self.use_location_loss:
                    output_target.append(locations)

                gan_losses = self.gan_model.train_on_batch(specular_real, output_target)

                gan_loss = gan_losses[0]

                print('Epoch: %d  Step: %d Discriminator loss (target): [%.3f] Discriminator loss (Specular): [%.3f] '
                      'Generator loss: [%.3f]' % (epoch+1, steps + 1, dis_loss_real, dis_loss_gen, gan_loss))

                self.network_data['epochs'] = epoch
                self.network_data['steps'] = steps

                if steps % 4 == 0:
                    with open(self.network_data_location, 'w') as f:
                        json.dump(self.network_data, f)

                    if gui is not None:
                        progress = int(((steps + (epoch - total_epochs)*data_len)/total_steps)*100)
                        gui.ids.train_progress_value.text = f"{progress}% Progress"
                        gui.ids.train_progress_bar.value = progress
                        gui.ids.train_progress_file.text = f"Total Epoch: {epoch}    Trained epochs: " \
                                                           f"{epoch - total_epochs}/{epochs}    Step: {steps}"
                        if self.use_vgg_loss:
                            gui.ids.train_progress_file.text += "  Using VGG Loss"
                        if self.use_location_loss:
                            gui.ids.train_progress_file.text += "  Using location Loss"

                        if gui.kill_signal:
                            break

                if steps % 100 == 0:
                    self.training_summary(self.generator, self.summary_location, epoch, steps, self.batch_size)
                    self.generator.save(self.network_location + 'generator.h5')
                    self.discriminator.save(self.network_location + 'discriminator.h5')

                steps += 1

            self.image_dataset.on_epoch_end()

            steps = 0
            epochs += 1

            self.training_summary(self.generator, self.summary_location, epoch, self.batch_size)
            self.generator.save(self.network_location + 'generator.h5')
            self.discriminator.save(self.network_location + 'discriminator.h5')

            self.network_data['epochs'] = epoch
            self.network_data['steps'] = steps
            with open(self.network_data_location, 'w') as f:
                json.dump(self.network_data, f)

            if gui is not None:
                if gui.kill_signal:
                    break

    def get_output(self, views):

        if not len(views) == self.image_pairs:
            raise ValueError(f"Expected {self.image_pairs} images. Received {len(views)} images instead.")

        input_images = []

        for view in views:
            hsv_image = self.image_dataset.convert_image_to_hsv(view, specular=True, gray=False)
            input_images.append(self.image_dataset.prepare_input(hsv_image, expand=True))

        input_image = np.concatenate(input_images, axis=2)

        generated_image = self.generator.predict(input_image)
        translated_image = ImageDataSet.translate_image(generated_image)

        div_point = int(translated_image.shape[1] / self.image_pairs)

        output_images = []

        for x in range(self.image_pairs):
            point = x * div_point
            output_images.append(np.uint8(translated_image[:, point:point+div_point, :]))

        return output_images

    def test_generator(self):

        images, _ = self.image_dataset.get_test_pair()

        image_1, image_2 = images

        return self.get_output([image_1, image_2])
