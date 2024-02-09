import cv2
import os
from  splice import subtract_images
def split_and_save_video_frames(video_path):
    # Проверка наличия папки spliced_video и создание ее, если она не существует
    output_dir = 'spliced_video'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Открыть видеофайл
    cap = cv2.VideoCapture(video_path)

    # Проверить, открыт ли видеофайл
    if not cap.isOpened():
        print("Ошибка открытия видеофайла.")
        return

    # Определить общее количество кадров и FPS видео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Определить размеры кадра
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Разделить видео на 4 равные части
    part_width = width // 2
    part_height = height // 2

    # Создать папку для сохранения изображений
    output_dir = 'spliced_video'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Счетчик для именования изображений
    count = 0

    # Чтение и сохранение кадров
    while True:
        ret, frame = cap.read()

        if ret:
            # Разделить кадр на четыре равные части
            top_left = frame[:part_height, :part_width]
            top_right = frame[:part_height, part_width:]
            bottom_left = frame[part_height:, :part_width]
            bottom_right = frame[part_height:, part_width:]

            # Сохранить каждую часть в папку spliced_video
            cv2.imwrite(os.path.join(output_dir, f'top_left_{count}.jpg'), top_left)
            cv2.imwrite(os.path.join(output_dir, f'top_right_{count}.jpg'), top_right)
            cv2.imwrite(os.path.join(output_dir, f'bottom_left_{count}.jpg'), bottom_left)
            cv2.imwrite(os.path.join(output_dir, f'bottom_right_{count}.jpg'), bottom_right)

            # Увеличить счетчик
            count += 1

        else:
            break

    # Освободить объект VideoCapture
    cap.release()
    subtract_images()


