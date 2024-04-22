import math
import tensorflow as tf
import numpy as np
import cv2

interpreter = tf.lite.Interpreter(model_path='3.tflite')
interpreter.allocate_tensors()


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


""" Возвращает угол между двумя векторами """
def angle_between(x1, y1, x2, y2):
    def scalar(x1, y1, x2, y2):
        return x1 * x2 + y1 * y2

    def module(x, y):
        return math.sqrt(x ** 2 + y ** 2)

    cos = scalar(x1, y1, x2, y2) / (module(x1, y1) * module(x2, y2))
    if -1 <= cos <= 1:
        return math.degrees(math.acos(cos))
    return 0


# TODO: соотнести ребра с частями тела
EDGES = {
    (0, 1): '1',
    (0, 2): '2',
    (1, 3): '3',
    (2, 4): '4',
    (0, 5): '5',
    (0, 6): '6',
    (5, 7): '7',
    (7, 9): '8',
    (6, 8): '9',
    (8, 10): '10',
    (5, 6): '11',
    (5, 11): '12',
    (6, 12): '13',
    (11, 12): '14',
    (11, 13): '15',
    (13, 15): '16',
    (12, 14): '17',
    (14, 16): '18'
}


def draw_connections(frame, keypoints, edges, confidence_threshold, edges_vec, edges_deg):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            # образуем новый вектор по координатам
            new_vec = [int(x2) - int(x1), int(y2) - int(y1)]
            # предыдущий вектор берем из словаря
            last_vec = edges_vec[color] if (color in edges_vec) else new_vec
            # считаем угол и записываем его в массив словаря
            deg = angle_between(last_vec[0], last_vec[1], new_vec[0], new_vec[1])
            edges_deg[color].append(deg)
            print("edge:", color, ",", "coords: (", int(x1), ",", int(y1), ";", int(x2), ",", int(y2), ")", "deg:", deg)
            # обновляем словарь векторов
            edges_vec[color] = new_vec
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


cap = cv2.VideoCapture(0)
EDGES_VEC = {}

EDGES_DEG = {}
for value in EDGES.values():
    EDGES_DEG[value] = [0]

while cap.isOpened():
    ret, frame = cap.read()

    # Reshape image
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
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4, EDGES_VEC, EDGES_DEG)
    draw_keypoints(frame, keypoints_with_scores, 0.4)

    cv2.imshow('MoveNet Lightning', frame)

    if cv2.waitKey(10) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
