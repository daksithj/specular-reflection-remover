from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, RiseInTransition, ScreenManagerException
from Interface.generate_interface import GenerateWindow
from Interface.train_interface import ModelWindow
from Interface.use_interface import UseWindow


class StartWindow(Screen):

    def __init__(self, **kwargs):

        super(StartWindow, self).__init__(**kwargs)
        self.generate_window = GenerateWindow()
        self.model_window = ModelWindow()
        self.use_window = UseWindow()

    def on_generate_window(self):

        try:
            self.manager.get_screen("generate_window")
        except ScreenManagerException:
            self.manager.add_widget(self.generate_window)

        self.manager.current = "generate_window"

    def on_model_window(self):

        try:
            self.manager.get_screen("model_window")
        except ScreenManagerException:
            self.manager.add_widget(self.model_window)

        self.manager.current = "model_window"

    def on_use_window(self):

        try:
            self.manager.get_screen("use_window")
        except ScreenManagerException:
            self.manager.add_widget(self.use_window)

        self.manager.current = "use_window"


class WindowManager(ScreenManager):

    def __init__(self, **kwargs):

        super(WindowManager, self).__init__(**kwargs)

        self.transition = RiseInTransition()
        self.add_widget(StartWindow())

        self.current = 'start_window'


class SpecToPoseApp(App):

    def build(self):
        self.title = 'SpecToPose'
        self.icon = 'Interface/logo.png'
        Builder.load_file('Interface/start_interface.kv')
        Builder.load_file('Interface/generate_interface.kv')
        Builder.load_file('Interface/train_interface.kv')
        Builder.load_file('Interface/use_interface.kv')
        wm = WindowManager()
        return wm


if __name__ == '__main__':
    SpecToPoseApp().run()
