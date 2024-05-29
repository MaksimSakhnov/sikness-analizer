import json

from kivy_garden.matplotlib import FigureCanvasKivyAgg
from kivymd.app import MDApp
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty, NumericProperty
from kivy.uix.button import Button
from kivymd.uix.button import MDFlatButton, MDRoundFlatIconButton
from kivy.uix.label import Label
from kivy.storage.jsonstore import JsonStore
import time



import cv2
import os

from analyze import split_and_save_video_frames
from pose import draw_plots

# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p": (640, 480),
}

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}







store = JsonStore('data.json')

class Loader(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (None, None)
        self.size = (50, 50)

        self.loader_image = Image(source='loader.gif',
                                  size_hint=(None, None),
                                  size=self.size,
                                  allow_stretch=True,
                                  keep_ratio=True)
        self.add_widget(self.loader_image)


class KivyCamera(BoxLayout):
    filename = StringProperty('video.avi')
    frames_per_second = NumericProperty(30.0)
    video_resolution = StringProperty('480p')

    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.duration = 10
        self.start_time = -1
        self.container = BoxLayout()
        self.render_start_screen()

    def render_start_screen(self, instance=None):
        if self.container:
            self.remove_widget(self.container)
        self.container = BoxLayout(orientation='vertical', spacing=50)
        self.grid = BoxLayout(orientation='horizontal', size_hint=(1, .10))
        self.header = BoxLayout(orientation='horizontal', size_hint=(1, .10))
        self.header_layot = AnchorLayout(anchor_x='right', anchor_y='top')
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

        self.header_layot.add_widget(self.history_button)
        self.header.add_widget(self.header_layot)

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

    def start_analyze(self, instance):
        self.remove_widget(self.container)
        self.loader.visible = True
        split_and_save_video_frames('video.avi')
        self.loader.visible = False
        self.container = BoxLayout(orientation='vertical', spacing=50)
        back_button = Button(text="Назад")
        back_button.bind(on_press=self.render_start_screen)
        self.container.add_widget(back_button)
        self.add_widget(self.container)

        row1 = BoxLayout(orientation='horizontal', spacing=20)
        row2 = BoxLayout(orientation='horizontal', spacing=20)
        row3 = BoxLayout(orientation='horizontal', spacing=20)
        row4 = BoxLayout(orientation='horizontal', spacing=20)

        self.res_1 = Image(source='result_photo/result1.png')
        self.res_2 = Image(source='result_photo/result2.png')
        self.res_3 = Image(source='result_photo/result3.png')
        self.res_4 = Image(source='result_photo/result4.png')

        row1.add_widget(Label(text='Lorem'))
        row1.add_widget(self.res_1)

        row2.add_widget(Label(text='Lorem'))
        row2.add_widget(self.res_2)

        row3.add_widget(Label(text='Lorem'))
        row3.add_widget(self.res_3)

        row4.add_widget(Label(text='Lorem'))
        row4.add_widget(self.res_4)

        self.container.add_widget(row1)
        self.container.add_widget(row2)
        self.container.add_widget(row3)
        self.container.add_widget(row4)

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

    def render_plot(self, instance):
        if self.container:
            self.remove_widget(self.container)
        # Чтение JSON файла
        with open('dict.json', 'r') as json_file:
            loaded_dict_with_str_keys = json.load(json_file)

        self.container = BoxLayout(orientation='vertical', spacing=50)
        # Преобразование строк ключей обратно в пары
        loaded_dict_with_tuple_keys = {tuple(map(int, k.strip('()').split(', '))): v for k, v in
                                       loaded_dict_with_str_keys.items()}

        canvas = FigureCanvasKivyAgg(draw_plots(loaded_dict_with_tuple_keys).gcf())
        self.container.add_widget(canvas)
        self.add_widget(self.container)


class CamApp(MDApp):
    def build(self):
        return KivyCamera()


if __name__ == '__main__':
    CamApp().run()



