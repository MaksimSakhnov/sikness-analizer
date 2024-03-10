import cv2
import numpy as np
import tensorflow as tf

# Загрузка модели PoseNet
interpreter = tf.lite.Interpreter(model_path='posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite')
interpreter.allocate_tensors()

# Получение индексов входных и выходных тензоров
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Функция для обработки изображения и обнаружения ключевых точек
def detect_keypoints(image):
    input_image = cv2.resize(image, (257, 257))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype(np.float32) / 255.0  # Нормализация
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    return keypoints

# Загрузка изображения
image_path = 'images/frame_1.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Обнаружение ключевых точек на изображении
keypoints = detect_keypoints(image_rgb)

# Пример анализа показателей движения
# Здесь можно реализовать код для отслеживания движения и анализа показателей
# Например, можно вычислить скорость движения руки на основе изменения положения ключевых точек со временем

# Отображение изображения с ключевыми точками
for keypoint in keypoints[0]:
    print(keypoint)
    y, x = keypoint[:2]  # Используйте только первые два значения
    cv2.circle(image, (int(x[0] * image.shape[1]), int(y[0] * image.shape[0])), 5, (0, 255, 0), -1)

print('1')
# Отображение изображения с ключевыми точками и информацией о движении
cv2.imshow('Pose Estimation', image)
print('2')
cv2.waitKey(0)
cv2.destroyAllWindows()
