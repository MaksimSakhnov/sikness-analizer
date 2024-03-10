import cv2
import os
from  splice import subtract_images
import cv2
import os

def split_and_save_video_frames(video_path, output_dir):
    # Проверка наличия папки для сохранения кадров
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Открыть видеофайл
    cap = cv2.VideoCapture(video_path)

    # Проверить, открыт ли видеофайл
    if not cap.isOpened():
        print("Ошибка открытия видеофайла.")
        return

    # Счетчик для именования кадров
    count = 0

    # Чтение и сохранение кадров
    while True:
        ret, frame = cap.read()

        if ret:
            # Сохранить кадр в папку
            cv2.imwrite(os.path.join(output_dir, f'frame_{count}.jpg'), frame)

            # Увеличить счетчик
            count += 1

        else:
            break

    # Освободить объект VideoCapture
    cap.release()

# Пример использования функции
video_path = 'video.mp4'
output_dir = './images/'
split_and_save_video_frames(video_path, output_dir)



