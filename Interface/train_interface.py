from kivy.uix.screenmanager import Screen, ScreenManagerException
from threading import Thread
from DatasetCreator.gen_tools import start_generating
import psutil
import shutil


class ModelWindow(Screen):

    def __init__(self, **kwargs):

        super(ModelWindow, self).__init__(**kwargs)
        # self.generate_window = GenerateWindow()
        # self.generate_window = GenerateWindow()

    def on_train(self):

        try:
            self.manager.get_screen("train_window")
        except ScreenManagerException:
            self.manager.add_widget(self.train_window)

        self.manager.current = "train_window"

    def on_generate(self):

        try:
            self.manager.get_screen("generate_window")
        except ScreenManagerException:
            self.manager.add_widget(self.generate_window)

        self.manager.current = "generate_window"
