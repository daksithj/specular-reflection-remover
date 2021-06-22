from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen, RiseInTransition, ScreenManagerException
from kivy.uix.popup import Popup
from threading import Thread
from DatasetCreator.gen_tools import start_generating
import psutil
import shutil
import numpy as np
import os

class GeneratePendingWindow(Screen):

    def __init__(self, **kwargs):

        super(GeneratePendingWindow, self).__init__(**kwargs)
        self.kill_signal = False
        self.params = {}


    def on_pre_enter(self, *args):

        self.ids.generate_progress_status.text = "Please wait until the data set is generated"
        self.ids.generate_progress_value.text = "0% Progress"
        self.ids.generate_progress_bar.value = 0
        self.ids.generate_progress_file.text = "Processing"
        self.ids.generate_finish_button.disabled = True
        self.ids.generate_cancel_button.disabled = False

    def on_enter(self, *args):
        process = start_generating(self.params['name'], self.params['models'], self.params['data_num'],
                                   self.params['background'], self.params['objects'])

        Thread(target=self.update_pending, args=[process], daemon=True).start()

    def update_pending(self, process):
        while True:
            line = process.stdout.readline()
            if not line:
                break
            if self.kill_signal:
                process.kill()
                break
            if "Generated output" in str(line):
                number = str(line).split('_')[1]
                number = int(number)
                progress = int(number/self.params['data_num']* 100)
                self.ids.generate_progress_bar.value = progress
                self.ids.generate_progress_value.text = f"{progress}% Progress"
                self.ids.generate_progress_file.text = f"Generated pair {number}/{self.params['data_num']}"

        self.ids.generate_finish_button.disabled = False
        shutil.rmtree(f"DatasetCreator/assets/{self.params['name']}")

    def cancel_process(self):
        self.kill_signal = True
        self.ids.generate_cancel_button.disabled = True
        self.ids.generate_progress_status.text = "Dataset generation cancelled!"
        self.ids.generate_progress_value.text = f"Cancelled"

class GenerateSettingsWindow(Screen):

    def __init__(self, **kwargs):

        super(GenerateSettingsWindow, self).__init__(**kwargs)

        self.custom_back_colors = []
        self.custom_object_colors = []
        self.random_back_color = False
        self.random_object_color = True
        self.file_list = []
        self.pending_window = GeneratePendingWindow()

    def on_pre_enter(self, *args):
        self.custom_back_colors = [self.ids.background_red, self.ids.background_green,
                              self.ids.background_blue, self.ids.background_alpha]

        self.custom_object_colors = [self.ids.object_red, self.ids.object_green,
                                   self.ids.object_blue, self.ids.object_alpha]

        self.ids.generate_dataset_name.text = ""
        self.ids.generate_dataset_number.text = ""

        self.ids.back_default.active = True
        self.ids.object_random.active = True

        for color in self.custom_back_colors:
            color.text = "0.03"
            color.disabled = True

        for color in self.custom_object_colors:
            color.text = "0.00"
            color.disabled = True

        self.random_back_color = False
        self.random_object_color = True

    def on_back_default(self):

        for selector in self.custom_back_colors:
            selector.text = "0.03"
            selector.disabled = True
            self.random_back_color = False

    def on_back_random(self):

        for selector in self.custom_back_colors:
            selector.text = "0.03"
            selector.disabled = True
            self.random_back_color = True

    def on_back_custom(self):

        for selector in self.custom_back_colors:
            selector.text = "0.03"
            selector.disabled = False
            self.random_back_color = False

    def on_object_random(self):

        for selector in self.custom_object_colors:
            selector.text = "0.00"
            selector.disabled = True
            self.random_object_color = True

    def on_object_custom(self):

        for selector in self.custom_object_colors:
            selector.text = "0.00"
            selector.disabled = False
            self.random_object_color = False

    def validate(self):

        if self.ids.generate_dataset_name.text == "":
            self.ids.generate_start_button.disabled = True
            return

        invalid_character = ['/', ">", "<", ":", "?", "*"]

        for character in invalid_character:
            if character in self.ids.generate_dataset_name.text:
                self.ids.generate_start_button.disabled = True
                return

        number = self.ids.generate_dataset_number.text

        try:
            number = int(number)
            if not (0 <= number < 100000):
                self.ids.generate_start_button.disabled = True
                return

        except ValueError:
            self.ids.generate_start_button.disabled = True
            return

        for back_color in self.custom_back_colors:
            try:
                color = float(back_color.text)
                if not (0 <= color <= 1.0):
                    self.ids.generate_start_button.disabled = True
                    return
            except ValueError:
                self.ids.generate_start_button.disabled = True
                return

        for object_color in self.custom_object_colors:
            try:
                color = float(object_color.text)
                if not (0 <= color <= 1.0):
                    self.ids.generate_start_button.disabled = True
                    return
            except ValueError:
                self.ids.generate_start_button.disabled = True
                return

        self.ids.generate_start_button.disabled = False
        return

    def on_start_generating(self):

        dataset_name = self.ids.generate_dataset_name.text
        dataset_num = int(self.ids.generate_dataset_number.text)

        if self.random_back_color:
            back_color = None
        else:
            back_color_list = []
            for color in self.custom_back_colors:
                back_color_list.append(color.text)
            back_color = ' '.join(back_color_list)

        if self.random_object_color:
            object_color = None
        else:
            object_color_list = []
            for color in self.custom_object_colors:
                object_color_list.append(color.text)
            object_color = ' '.join(object_color_list)

        try:
            self.manager.get_screen('generate_pending_window')
        except ScreenManagerException:
            self.manager.add_widget(self.pending_window)

        self.pending_window.params = {
            'name': dataset_name,
            'data_num': dataset_num,
            'models': self.file_list,
            'background': back_color,
            'objects': object_color
        }

        self.manager.current = 'generate_pending_window'

    def on_back(self):
        self.manager.current = "generate_window"



class GenerateWindow(Screen):

    def __init__(self, **kwargs):

        super(GenerateWindow, self).__init__(**kwargs)
        self.file_selection = []
        self.file_list = []
        self.settings_window = GenerateSettingsWindow()

    def on_pre_enter(self, *args):

        self.ids.model_file_chooser.path = '.'
        self.ids.model_file_chooser_drive.text = 'Choose drive'
        self.file_selection = []
        self.file_list = []
        self.ids.generate_add_button.disabled = True
        self.ids.generate_remove_button.disabled = True
        self.ids.generate_submit_button.disabled = True
        self.update_file_num()

    def update_drives(self):

        drive_list = []
        disk_partitions = psutil.disk_partitions(all=True)

        for partition in disk_partitions:
            drive_list.append(partition.device)

        self.ids.model_file_chooser_drive.values = drive_list

    def update_file_path_dir(self):

        drive = self.ids.index_file_chooser_drive.text
        if drive == 'Choose drive':
            self.ids.index_file_chooser.path = '.'
        else:
            self.ids.index_file_chooser.path = drive

    def on_select_file(self, file_name):

        try:
            self.file_selection = file_name

            if len(self.file_selection) > 0:
                self.ids.generate_add_button.disabled = False
            else:
                self.ids.generate_add_button.disabled = True
        except IndexError:
            pass

    def on_add_file(self):

        if not len(self.file_selection) == 0:
            self.file_list.extend(self.file_selection)
            self.ids.generate_remove_button.disabled = False
            self.ids.model_file_chooser.selection = []
            self.ids.generate_submit_button.disabled = False
        self.update_file_num()

    def on_remove_files(self):

        if len(self.file_list) > 0:
            self.file_list.pop()

        if len(self.file_list) == 0:
            self.ids.generate_remove_button.disabled = True
            self.ids.generate_submit_button.disabled = True

        self.update_file_num()

    def update_file_num(self):
        self.ids.generate_file_num.text = f'Files added: {len(self.file_list)}'

    def on_submit_models(self):
        try:
            self.manager.get_screen('generate_settings_window')
        except ScreenManagerException:
            self.manager.add_widget(self.settings_window)

        self.manager.current = 'generate_settings_window'
        self.settings_window.file_list = self.file_list[:]

    def on_back(self):
        self.manager.current = "start_window"


class StartWindow(Screen):

    def __init__(self, **kwargs):

        super(StartWindow, self).__init__(**kwargs)
        self.generate_window = GenerateWindow()
        #elf.generate_window = GenerateWindow()

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


class WindowManager(ScreenManager):

    def __init__(self, **kwargs):

        super(WindowManager, self).__init__(**kwargs)

        self.transition = RiseInTransition()
        self.add_widget(StartWindow())
        self.add_widget(GenerateWindow())
        # self.add_widget(GenerateSettingsWindow())
        self.current = 'generate_window'


class SpecToPoseApp(App):

    def build(self):
        self.title = 'SpecToPose'
        self.icon = 'Interface/logo.png'
        Builder.load_file('Interface/interface.kv')
        wm = WindowManager()
        return wm


if __name__ == '__main__':
    SpecToPoseApp().run()
