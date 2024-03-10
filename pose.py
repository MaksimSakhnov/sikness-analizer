import tensorflow as tf
import numpy as np
import cv2

# Загрузка модели PoseNet
model = tf.keras.models.load_model('posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite')

# Функция для обработки изображения
def process_image(image):
    input_image = tf.convert_to_tensor(image, dtype=tf.float32)
    input_image = tf.image.resize(input_image, (257, 257))
    input_image = input_image[tf.newaxis, ...]
    return input_image

# Загрузка изображения
image_path = 'person.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Обработка изображения
input_image = process_image(image_rgb)

# Получение предсказаний модели
outputs = model.predict(input_image)

# Отображение предсказаний
keypoints = outputs[0]['output_0']
for keypoint in keypoints:
    y, x, _ = keypoint
    cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), 5, (0, 255, 0), -1)

# Отображение изображения с оцененными позами
cv2.imshow('Pose Estimation', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
