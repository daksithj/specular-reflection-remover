from kivy.uix.screenmanager import Screen, ScreenManagerException
from kivy.uix.popup import Popup
from threading import Thread
from DeepLearning.gan import SpecToPoseNet
from DeepLearning.dataset_gan import ImageDataSet
import shutil
import json
import os

DATASET_FOLDER = "DatasetCreator/Output"
NETWORK_FOLDER = "DeepLearning/Networks"


def get_model_list():

    dir_list = []

    for dir_in_list in os.listdir(NETWORK_FOLDER):

        directory = NETWORK_FOLDER + '/' + dir_in_list

        if os.path.isdir(directory):

            dir_file_list = os.listdir(directory)

            if "network_data.json" in dir_file_list:
                dir_list.append(dir_in_list)

    return dir_list


class ModelWindow(Screen):

    def __init__(self, **kwargs):

        super(ModelWindow, self).__init__(**kwargs)
        self.new_model_window = NewModelWindow()
        self.train_model_window = TrainModelWindow()

    def on_train_model(self):

        try:
            self.manager.get_screen("train_model_window")
        except ScreenManagerException:
            self.manager.add_widget(self.train_model_window)

        self.manager.current = "train_model_window"

    def on_new_model(self):

        try:
            self.manager.get_screen("new_model_window")
        except ScreenManagerException:
            self.manager.add_widget(self.new_model_window)

        self.manager.current = "new_model_window"


class NewModelWindow(Screen):

    def __init__(self, **kwargs):

        super(NewModelWindow, self).__init__(**kwargs)
        self.folder_list = ["Diffuse", "Locations", "Matrix", "Specular"]
        self.overwrite = False
        self.success_popup = CreateSuccessPopup()

    def on_pre_enter(self, *args):
        self.overwrite = False
        self.ids.dataset_chooser.text = 'Choose a dataset'
        self.ids.new_model_button.disabled = True
        self.ids.new_model_name.text = ""

    def list_data_sets(self):

        dir_list = []

        for dir_in_list in os.listdir(DATASET_FOLDER):

            directory = DATASET_FOLDER + '/' + dir_in_list

            if os.path.isdir(directory):

                dir_file_list = os.listdir(directory)

                if set(dir_file_list) == set(self.folder_list):
                    dir_list.append(dir_in_list)

        self.ids.dataset_chooser.values = dir_list

    def validate(self):
        invalid_choices = ['', 'Choose a dataset']

        if self.ids.dataset_chooser.text in invalid_choices:
            return

        if self.ids.new_model_name.text == "":
            self.ids.new_model_button.disabled = True
            return

        invalid_character = ['/', ">", "<", ":", "?", "*", "."]

        for character in invalid_character:
            if character in self.ids.new_model_name.text:
                self.ids.new_model_button.disabled = True
                return

        model_list = get_model_list()

        if self.ids.new_model_name.text in model_list:
            self.ids.new_model_button.text = "Overwrite model"
            self.overwrite = True
        else:
            self.ids.new_model_button.text = "Create model"
            self.overwrite = False

        self.ids.new_model_button.disabled = False

    def on_create_model(self):

        model_name = self.ids.new_model_name.text

        if self.overwrite:
            shutil.rmtree(f"{NETWORK_FOLDER}/{model_name}")

        dataset = ImageDataSet(self.ids.dataset_chooser.text)
        _ = SpecToPoseNet(model_name, dataset)
        self.success_popup.open()
        self.manager.current = 'model_window'

    def on_back(self):
        self.manager.current = 'model_window'


class CreateSuccessPopup(Popup):

    def __init__(self, **kwargs):
        super(CreateSuccessPopup, self).__init__(**kwargs)


class TrainModelWindow(Screen):

    def __init__(self, **kwargs):
        super(TrainModelWindow, self).__init__(**kwargs)

        self.pending_window = TrainPendingWindow()
        self.epochs = -1
        self.use_vgg = -1
        self.use_pose = -1

    def on_pre_enter(self, *args):
        self.ids.train_model_chooser.text = 'Choose a model'
        self.ids.train_dataset_name.text = '-'
        self.ids.train_image_size.text = '-'
        self.ids.train_channels.text = '-'
        self.ids.train_pairs.text = '-'
        self.ids.train_model_epochs.text = '-'
        self.ids.train_steps.text = '-'

        self.ids.train_epochs.text = ""
        self.epochs = -1

        self.ids.use_vgg_switch.disabled = True
        self.ids.use_vgg_switch.active = False
        self.ids.use_vgg_epoch.disabled = True
        self.use_vgg = -1

        self.ids.use_pose_switch.disabled = True
        self.ids.use_pose_switch.active = False
        self.ids.use_pose_epoch.disabled = True
        self.use_pose = -1

        self.ids.train_start_button.disabled = True

    def list_models(self):
        self.ids.train_model_chooser.values = get_model_list()

    def on_selection(self):

        model_name = self.ids.train_model_chooser.text

        network_data_location = f"{NETWORK_FOLDER}/{model_name}/network_data.json"

        with open(network_data_location) as f:
            network_data = json.load(f)

        self.ids.train_dataset_name.text = network_data['dataset_name']
        self.ids.train_image_size.text = f"{network_data['image_size']} x {network_data['image_size']}"
        self.ids.train_channels.text = str(network_data['channels'])
        self.ids.train_pairs.text = str(network_data['pairs'])
        self.ids.train_model_epochs.text = str(network_data['epochs'])
        self.ids.train_steps.text = str(network_data['steps'])

        self.validate()

    def validate(self):

        try:
            epochs = int(self.ids.train_epochs.text)
            if not (0 < epochs < 5000):
                raise ValueError

            self.epochs = epochs
            self.ids.use_vgg_switch.disabled = False
            self.ids.use_pose_switch.disabled = False
        except ValueError:
            self.epochs = -1

            self.ids.use_vgg_switch.disabled = True
            self.ids.use_vgg_switch.active = False
            self.ids.use_vgg_epoch.disabled = True
            self.use_vgg = -1

            self.ids.use_pose_switch.disabled = True
            self.ids.use_pose_switch.active = False
            self.ids.use_pose_epoch.disabled = True
            self.use_pose = -1

            self.ids.train_start_button.disabled = True
            return

        if self.ids.use_vgg_switch.active:
            self.ids.use_vgg_epoch.disabled = False
        else:
            self.ids.use_vgg_epoch.disabled = True
            self.use_vgg = -1

        if self.ids.use_pose_switch.active:
            self.ids.use_pose_epoch.disabled = False
        else:
            self.ids.use_pose_epoch.disabled = True
            self.use_pose = -1

        if self.ids.use_vgg_switch.active:
            try:
                self.use_vgg = int(self.ids.use_vgg_epoch.text)
                if not (0 <= self.use_vgg < epochs):
                    raise ValueError
            except ValueError:
                self.use_vgg = -1
                self.ids.train_start_button.disabled = True
                return

        if self.ids.use_pose_switch.active:
            self.ids.use_pose_epoch.disabled = False
            try:
                self.use_pose = int(self.ids.use_pose_epoch.text)
                if not (0 <= self.use_pose < epochs):
                    raise ValueError
            except ValueError:
                self.use_pose = -1
                self.ids.train_start_button.disabled = True
                return

        model_name = self.ids.train_model_chooser.text
        invalid_choices = ['', 'Choose a model']

        if model_name in invalid_choices:
            self.ids.train_start_button.disabled = True
            return

        self.ids.train_start_button.disabled = False

    def start_train(self):
        try:
            self.manager.get_screen('train_pending_window')
        except ScreenManagerException:
            self.manager.add_widget(self.pending_window)

        self.pending_window.params = {
            'model_name': self.ids.train_model_chooser.text,
            'epochs': self.epochs,
            'use_vgg': self.use_vgg,
            'use_pose': self.use_pose
        }

        self.manager.current = 'train_pending_window'

    def on_back(self):
        self.manager.current = "model_window"


class TrainPendingWindow(Screen):

    def __init__(self, **kwargs):

        super(TrainPendingWindow, self).__init__(**kwargs)
        self.kill_signal = False
        self.params = {}

    def on_pre_enter(self, *args):
        self.ids.train_progress_status.text = "Please wait until the model is trained"
        self.ids.train_progress_bar.value = 0
        self.ids.train_progress_value.text = "0% Progress"
        self.ids.train_finish_button.disabled = True
        self.ids.train_cancel_button.disabled = False
        self.ids.train_progress_status.text = ""

    def on_enter(self, *args):
        Thread(target=self.start_training, args=[], daemon=True).start()

    def start_training(self):

        try:
            network = SpecToPoseNet(self.params['model_name'])

            network.train_gan(epochs=self.params['epochs'], toggle_vgg=self.params['use_vgg'],
                              toggle_location_loss=self.params['use_pose'], gui=self)

            if not self.kill_signal:
                self.ids.train_progress_status.text = "Training complete!"
                self.ids.train_progress_bar.value = 100
                self.ids.train_progress_value.text = "100% Progress"
            self.ids.train_finish_button.disabled = False
            self.ids.train_cancel_button.disabled = True
        except ValueError as e:

            self.ids.train_progress_status.text = "Encountered an error. Training cancelled!"
            self.ids.train_progress_file.text = str(e)

            self.ids.train_cancel_button.disabled = True
            self.ids.train_finish_button.disabled = False

    def cancel_process(self):
        self.kill_signal = True
        self.ids.train_progress_status.text = "Training cancelled by user!"
        self.ids.train_cancel_button.disabled = True
