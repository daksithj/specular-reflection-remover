from kivy.uix.screenmanager import Screen, ScreenManagerException
from kivy.graphics.texture import Texture
from DeepLearning.gan import SpecToPoseNet
from PoseEstimation.pose_estimator import get_pose, get_pose_with_centres, get_disparity_map_matrix
from Interface.train_interface import get_model_list
from threading import Thread
import psutil
import cv2


class UseWindow(Screen):

    def __init__(self, **kwargs):

        super(UseWindow, self).__init__(**kwargs)
        self.model = None
        self.file_selection = ""
        self.left_image = None
        self.right_image = None

        self.result_window = ResultWindow()

    def on_pre_enter(self, *args):

        self.ids.use_model_chooser.text = "Choose a model"
        self.ids.use_file_chooser_drive.text = "Choose drive"
        self.ids.use_file_chooser.path = "."
        self.ids.use_file_chooser.selection = []

        self.ids.left_remove_button.disabled = True
        self.ids.left_add_button.disabled = True

        self.ids.right_remove_button.disabled = True
        self.ids.right_add_button.disabled = True

        self.ids.convert_button.disabled = True

        self.model = None
        self.file_selection = ""
        self.left_image = None
        self.right_image = None

    def list_models(self):
        self.ids.use_model_chooser.values = get_model_list()

    def select_model(self):
        if not self.ids.use_model_chooser.text == "Choose a model":
            self.model = SpecToPoseNet(self.ids.use_model_chooser.text)
            self.pre_validate()

    def update_drives(self):

        drive_list = []
        disk_partitions = psutil.disk_partitions(all=True)

        for partition in disk_partitions:
            drive_list.append(partition.device)

        self.ids.use_file_chooser_drive.values = drive_list

    def update_file_path_dir(self):

        drive = self.ids.use_file_chooser_drive.text
        if drive == 'Choose drive':
            self.ids.use_file_chooser.path = '.'
        else:
            self.ids.use_file_chooser.path = drive

    def on_select_file(self, file_name):

        try:
            self.file_selection = file_name

            if len(self.file_selection) > 0:
                if self.left_image is None:
                    self.ids.left_add_button.disabled = False
                if self.right_image is None:
                    self.ids.right_add_button.disabled = False
            else:
                self.ids.left_add_button.disabled = True
                self.ids.right_add_button.disabled = True

        except IndexError:
            self.file_selection = ""

    def add_left_image(self):

        if len(self.file_selection) == 0:
            return

        self.left_image = cv2.imread(self.file_selection[0])
        self.file_selection = ""
        self.ids.use_file_chooser.selection = []
        self.ids.left_remove_button.disabled = False
        self.ids.left_add_button.disabled = True

        self.pre_validate()

    def remove_left_image(self):

        self.left_image = None
        self.ids.left_remove_button.disabled = True

        if not self.ids.use_file_chooser.selection == []:
            self.ids.left_add_button.disabled = False

        self.pre_validate()

    def add_right_image(self):

        if len(self.file_selection) == 0:
            return

        self.right_image = cv2.imread(self.file_selection[0])
        self.file_selection = ""
        self.ids.use_file_chooser.selection = []
        self.ids.right_remove_button.disabled = False
        self.ids.right_add_button.disabled = True

        self.pre_validate()

    def remove_right_image(self):

        self.right_image = None
        self.ids.right_remove_button.disabled = True

        if not self.ids.use_file_chooser.selection == []:
            self.ids.right_add_button.disabled = False

        self.pre_validate()

    def pre_validate(self):

        if self.model is None:
            self.ids.convert_button.disabled = True
            return

        if self.right_image is None or self.left_image is None:
            self.ids.convert_button.disabled = True
            return

        try:
            if not (self.right_image.shape[0] == self.right_image.shape[1]):
                raise ValueError("Right image should be square")

            if not (self.right_image.shape[0] == self.model.image_size):
                raise ValueError("Right image dimension doesnt match model dimensions")

            if not (self.right_image.shape[2] == self.model.channels):
                raise ValueError("Right image channels do not match model")

            if not (self.left_image.shape[0] == self.left_image.shape[1]):
                raise ValueError("Left image should be square")

            if not (self.left_image.shape[0] == self.model.image_size):
                raise ValueError("Left image dimension doesnt match model dimensions")

            if not (self.left_image.shape[2] == self.model.channels):
                raise ValueError("Left image channels do not match model")
        except ValueError as e:
            self.ids.convert_error.text = str(e)
            self.ids.convert_button.disabled = True
            return

        self.ids.convert_button.disabled = False

    def convert(self):
        try:
            self.manager.get_screen('result_window')
        except ScreenManagerException:
            self.manager.add_widget(self.result_window)

        self.result_window.model = self.model
        self.result_window.left_image = self.left_image
        self.result_window.right_image = self.right_image

        self.manager.current = "result_window"

    def on_back(self):
        self.manager.current = "start_window"


class ResultWindow(Screen):

    def __init__(self, **kwargs):

        super(ResultWindow, self).__init__(**kwargs)
        self.model = None
        self.left_image = None
        self.right_image = None

        self.convert_left = None
        self.convert_right = None

        self.location_set = []
        self.images_set = []

    def on_enter(self, *args):
        self.ids.result_chooser.text = 'Processing'
        self.ids.result_x.text = "-"
        self.ids.result_y.text = "-"
        self.ids.result_z.text = "-"
        self.ids.result_chooser.value = []

        self.convert_left, self.convert_right = self.model.get_output([self.left_image, self.right_image])

        buf1 = cv2.flip(self.convert_left, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(self.convert_left.shape[1], self.convert_left.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.result_image.texture = texture1

        Thread(target=self.calculate_pose, args=[], daemon=True).start()

    def calculate_pose(self):

        disparity_map_matrix = get_disparity_map_matrix(self.model.image_dataset.matrix_dir,
                                                        self.convert_left.shape[:2])

        poses, self.images_set = get_pose_with_centres(self.convert_left, self.convert_right, disparity_map_matrix)

        specular_pose_image = get_pose(self.left_image, self.right_image, disparity_map_matrix, draw_line=True,
                                       get_image=True)

        self.images_set.append(specular_pose_image)

        self.location_set = []
        result_set = []

        for pose in poses:
            self.location_set.append(pose['location'])

        for x in range(len(self.location_set)):
            result_set.append(f"Object {x}")

        result_set.append('Specular image')

        self.ids.result_chooser.values = result_set

        self.ids.result_x.text = "-"
        self.ids.result_y.text = "-"
        self.ids.result_z.text = "-"

        self.ids.result_chooser.text = 'Choose an object'

    def on_selection(self):
        if self.ids.result_chooser.text == 'Choose an object' or self.ids.result_chooser.text == 'Processing':
            return

        if self.ids.result_chooser.text == 'Specular image':
            self.ids.result_x.text = "-"
            self.ids.result_y.text = "-"
            self.ids.result_z.text = "-"

            frame = self.images_set[-1]
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.result_image.texture = texture1
            return

        object_name = self.ids.result_chooser.text

        object_num = int(object_name.split(" ")[1])

        frame = self.images_set[object_num]
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.result_image.texture = texture1

        pose = self.location_set[object_num]

        self.ids.result_x.text = str(pose[0])
        self.ids.result_y.text = str(pose[1])
        self.ids.result_z.text = str(pose[2])
