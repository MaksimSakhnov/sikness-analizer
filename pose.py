import tensorflow as tf
import numpy as np
import cv2
import os
from poseLib import draw_connections, draw_keypoints, EDGES, EDGES_VEC, EDGES_DEG, RESULT, RESULT_EDGES, draw_plots

videos_dir = 'videos'
plots_dir = 'plots'

interpreter = tf.lite.Interpreter(model_path='3.tflite')
interpreter.allocate_tensors()

if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)


def process_video(video_path, plot_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

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
        draw_connections(frame, keypoints_with_scores, EDGES, 0.4, EDGES_VEC, EDGES_DEG, RESULT, RESULT_EDGES)
        draw_keypoints(frame, keypoints_with_scores, 0.4)

        cv2.imshow('MoveNet Lightning', frame)

        key = cv2.waitKey(10)

        if key == ord('q') & 0xFF:
            break

    cap.release()
    cv2.destroyAllWindows()
    plt = draw_plots(RESULT_EDGES)
    plt.savefig(plot_path)
    plt.close()


for video_name in os.listdir(videos_dir):
    video_path = os.path.join(videos_dir, video_name)
    plot_name = os.path.splitext(video_name)[0] + '.png'  # Имя графика = имя видео + .png
    plot_path = os.path.join(plots_dir, plot_name)

    process_video(video_path, plot_path)
    print(f"Обработано видео: {video_name}, график сохранен в: {plot_path}")
