from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, RiseInTransition, ScreenManagerException
from Interface.generate_interface import GenerateWindow
from Interface.train_interface import ModelWindow, TrainModelWindow


class StartWindow(Screen):

    def __init__(self, **kwargs):

        super(StartWindow, self).__init__(**kwargs)
        self.generate_window = GenerateWindow()
        self.model_window = ModelWindow()

    def on_model_window(self):

        try:
            self.manager.get_screen("model_window")
        except ScreenManagerException:
            self.manager.add_widget(self.model_window)

        self.manager.current = "model_window"

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
        Builder.load_file('Interface/train_interface.kv')
        self.add_widget(TrainModelWindow())
        self.current = 'train_model_window'


class SpecToPoseApp(App):

    def build(self):
        self.title = 'SpecToPose'
        self.icon = 'Interface/logo.png'
        Builder.load_file('Interface/start_interface.kv')
        Builder.load_file('Interface/generate_interface.kv')
        Builder.load_file('Interface/train_interface.kv')
        wm = WindowManager()
        return wm


if __name__ == '__main__':
    SpecToPoseApp().run()
