import json
from kivy_garden.matplotlib import FigureCanvasKivyAgg
from kivymd.app import MDApp
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty, NumericProperty
from kivymd.uix.button import MDFlatButton, MDRoundFlatIconButton
from kivy.storage.jsonstore import JsonStore
import time
import cv2
import os
import tensorflow as tf
import numpy as np
from kivymd.icon_definitions import md_icons
import datetime
from poseLib import draw_connections, draw_keypoints, EDGES, EDGES_VEC, EDGES_DEG, RESULT, RESULT_EDGES, draw_plots

# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p": (640, 480),
}

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

interpreter = tf.lite.Interpreter(model_path='3.tflite')
interpreter.allocate_tensors()



store = JsonStore('data.json')

class Loader(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (None, None)
        self.size = (50, 50)

        self.loader_image = Image(source='./assets/loader.gif',
                                  size_hint=(None, None),
                                  size=self.size,
                                  allow_stretch=True,
                                  keep_ratio=True)
        self.add_widget(self.loader_image)


class KivyCamera(BoxLayout):
    frames_per_second = NumericProperty(30.0)
    video_resolution = StringProperty('480p')

    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.duration = 10
        self.start_time = -1
        self.container = BoxLayout()
        self.render_start_screen()

    def render_start_screen(self, instance=None):
        time = datetime.datetime.now().strftime("%m%d%Y%H%M%S")
        self.filename = 'video_' + time + '.mp4'
        if self.container:
            self.remove_widget(self.container)
        self.container = BoxLayout(orientation='vertical', spacing=50)
        self.grid = BoxLayout(orientation='horizontal', size_hint=(1, .10))
        self.header = BoxLayout(orientation='horizontal', size_hint=(1, .10))
        self.header_layout = AnchorLayout(anchor_x='right', anchor_y='top')
        self.footer_layout = AnchorLayout(anchor_x='center', anchor_y='center')

        self.img1 = Image()
        self.capture = cv2.VideoCapture(0)
        self.out = None
        self.recording = False
        self.record_button = MDRoundFlatIconButton(icon='camera', theme_text_color='ContrastParentBackground', text="Начать запись", pos_hint={'center_x': 0.5, 'center_y': 0.5}, size_hint=(None, None), size=(100, 50))
        self.history_button = MDFlatButton(text='История', pos_hint={'center_x': 0.5, 'center_y': 0.5}, size_hint=(None, None), size=(100, 50))

        self.loader = Loader()
        self.loader.visible = False

        self.record_button.bind(on_release=self.toggle_recording)
        self.history_button.bind(on_release=self.render_plot)

        self.container.add_widget(self.header)
        self.container.add_widget(self.img1)
        self.container.add_widget(self.grid)

        self.header_layout.add_widget(self.history_button)
        self.header.add_widget(self.header_layout)

        self.footer_layout.add_widget(self.record_button)
        self.grid.add_widget(self.footer_layout)

        self.add_widget(self.container)
        Clock.schedule_interval(self.update, 1 / self.frames_per_second)

    def update(self, *args):
        ret, frame = self.capture.read()
        if ret:
            if self.recording:
                self.out.write(frame)
                remaining_time = self.duration - (time.time() - self.start_time)
                if remaining_time <= 0:
                    self.toggle_recording(self)
                else:
                    self.record_button.text = str(int(remaining_time))
                    img = frame.copy()
                    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
                    input_image = tf.cast(img, dtype=tf.float32)

                    # Setup input and output
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()

                    # Make predictions
                    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
                    interpreter.invoke()
                    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

                    # Rendering
                    draw_connections(frame, keypoints_with_scores, EDGES, 0.4, EDGES_VEC, EDGES_DEG, RESULT,
                                     RESULT_EDGES)
                    draw_keypoints(frame, keypoints_with_scores, 0.4)
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
            texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
            self.img1.texture = texture

    def toggle_recording(self, instance):
        print('toggle')
        if not self.recording:
            self.start_time = time.time()
            self.recording = True
            self.out = cv2.VideoWriter(self.filename, self.get_video_type(self.filename), self.frames_per_second,
                                       self.get_dims(self.capture, self.video_resolution))
            self.record_button.text = "Остановить запись"
        else:
            self.start_time = -1
            self.recording = False
            self.out.release()
            self.record_button.text = "Начать запись"
            self.render_plot()

    def change_resolution(self, cap, width, height):
        cap.set(3, width)
        cap.set(4, height)

    # grab resolution dimensions and set video capture to it.
    def get_dims(self, cap, video_resolution='1080p'):
        width, height = STD_DIMENSIONS["480p"]
        if self.video_resolution in STD_DIMENSIONS:
            width, height = STD_DIMENSIONS[self.video_resolution]
        ## change the current capture device
        ## to the resulting resolution
        self.change_resolution(cap, width, height)
        return width, height

    def get_video_type(self, filename):
        filename, ext = os.path.splitext(filename)
        if ext in VIDEO_TYPE:
            return VIDEO_TYPE[ext]
        return VIDEO_TYPE['avi']

    def render_plot(self):
        if self.container:
            self.remove_widget(self.container)
        # # Чтение JSON файла
        # with open('dict.json', 'r') as json_file:
        #     loaded_dict_with_str_keys = json.load(json_file)
        #
        # self.container = BoxLayout(orientation='vertical', spacing=50)
        # # Преобразование строк ключей обратно в пары
        # loaded_dict_with_tuple_keys = {tuple(map(int, k.strip('()').split(', '))): v for k, v in
        #                                loaded_dict_with_str_keys.items()}

        self.header = BoxLayout(orientation='horizontal', size_hint=(1, .10))
        self.header_layout = AnchorLayout(anchor_x='left', anchor_y='top')

        self.back_button = MDRoundFlatIconButton(icon='keyboard-backspace', theme_text_color='ContrastParentBackground', text="Назад", pos_hint={'center_x': 0.5, 'center_y': 0.5}, size_hint=(None, None), size=(100, 50))
        self.back_button.bind(on_release=self.render_start_screen)


        self.container = BoxLayout(orientation='vertical', spacing=50)
        canvas = FigureCanvasKivyAgg(draw_plots(RESULT_EDGES).gcf())

        self.container.add_widget(self.header)
        self.container.add_widget(canvas)

        self.header_layout.add_widget(self.back_button)
        self.header.add_widget(self.header_layout)

        self.add_widget(self.container)


class CamApp(MDApp):
    def build(self):
        return KivyCamera()


if __name__ == '__main__':
    CamApp().run()



