from kivy.uix.screenmanager import Screen, ScreenManagerException
from kivy.uix.popup import Popup
from threading import Thread
from DeepLearning.gan import SpecToPoseNet
from DeepLearning.dataset_gan import ImageDataSet
import psutil
import shutil
import json
import os

DATASET_FOLDER = "DatasetCreator/Output"
NETWORK_FOLDER = "DeepLearning/Networks"

def get_model_list():

    dir_list = []

    for dir_in_list in os.listdir(NETWORK_FOLDER):

        directory = NETWORK_FOLDER  + '/' + dir_in_list

        if os.path.isdir(directory):

            dir_file_list = os.listdir(directory)

            if "network_data.json" in dir_file_list:
                dir_list.append(dir_in_list)

    return dir_list

class ModelWindow(Screen):

    def __init__(self, **kwargs):

        super(ModelWindow, self).__init__(**kwargs)
        self.new_model_window = NewModelWindow()
        # self.generate_window = GenerateWindow()

    def on_train(self):

        try:
            self.manager.get_screen("train_window")
        except ScreenManagerException:
            self.manager.add_widget(self.train_window)

        self.manager.current = "train_window"

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
