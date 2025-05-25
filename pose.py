import tensorflow as tf
import numpy as np
import cv2
import os

from kalmanFilter import KalmanFilter2D, MovingAverageFilter2D, ExponentialSmoothingFilter2D, MedianFilter2D, CombinedMedianMovingAverage
from poseLib import draw_connections, draw_keypoints, compute_metrics, plot_trajectory, draw_center_of_mass, draw_plots, \
    compute_stability_score, save_edges_to_json

videos_dir = 'videos'
plots_dir = 'plots_centered'
analyze_dir = 'analyze'
elipsis_dir = 'elipsis'

interpreter = tf.lite.Interpreter(model_path='3.tflite')
interpreter.allocate_tensors()

if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)


def process_video(video_path, plot_path, elipsis_path, analyze_path):
    cap = cv2.VideoCapture(video_path)
    kf_filter = KalmanFilter2D()
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

    EDGES_VEC = {}
    RESULT = {
        0: [[], []],
        1: [[], []],
        2: [[], []],
        3: [[], []],
        4: [[], []],
        5: [[], []],
        6: [[], []],
        7: [[], []],
        8: [[], []],
        9: [[], []],
        10: [[], []],
        11: [[], []],
        12: [[], []],
        13: [[], []],
        14: [[], []],
        15: [[], []],
        16: [[], []],
    }

    RESULT_EDGES = {
        (0, 1): [[], []],
        (0, 2): [[], []],
        (1, 3): [[], []],
        (2, 4): [[], []],
        (0, 5): [[], []],
        (0, 6): [[], []],
        (5, 7): [[], []],
        (7, 9): [[], []],
        (6, 8): [[], []],
        (8, 10): [[], []],
        (5, 6): [[], []],
        (5, 11): [[], []],
        (6, 12): [[], []],
        (11, 12): [[], []],
        (11, 13): [[], []],
        (13, 15): [[], []],
        (12, 14): [[], []],
        (14, 16): [[], []]
    }

    EDGES_DEG = {}
    for value in EDGES.values():
        EDGES_DEG[value] = [0]

    keypoint_filters = [KalmanFilter2D() for _ in range(17)]

    CENTER_COORDS = []

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

        shaped = np.squeeze(np.multiply(keypoints_with_scores[0], [frame.shape[0], frame.shape[1], 1]))

        # Фильтрация каждого ключевого узла
        for i, (y_kp, x_kp, conf) in enumerate(shaped):
            if conf > 0.4:
                x_filtered, y_filtered = keypoint_filters[i].update((x_kp, y_kp))
                shaped[i][0] = y_filtered
                shaped[i][1] = x_filtered
            else:
                keypoint_filters[i].update(None)

        # Rendering
        draw_connections(frame, shaped, EDGES, 0.4, EDGES_VEC, EDGES_DEG, RESULT, RESULT_EDGES)
        draw_keypoints(frame, shaped, 0.4)
        draw_center_of_mass(frame, shaped, 0.4, kf_filter, CENTER_COORDS)

        cv2.imshow('MoveNet Lightning', frame)

        key = cv2.waitKey(10)

        if key == ord('q') & 0xFF:
            break

    cap.release()
    cv2.destroyAllWindows()
    metrics = compute_metrics(CENTER_COORDS)
    stability_score, predicted_class = compute_stability_score(metrics)
    print('Оценка по центру масс: Score = ' + str(stability_score) + " Предсказанный класс: " + str(predicted_class) + "\n")
    with open(analyze_path, 'w+', encoding='utf-8') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    elipsis = plot_trajectory(CENTER_COORDS)
    elipsis.savefig(elipsis_path, bbox_inches='tight')
    # save_edges_to_json(RESULT_EDGES, //path)
    plt = draw_plots(RESULT_EDGES)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()



video_to_process = input("Введите имя видео для обработки (оставьте пустым для обработки всех): ").strip()

if video_to_process:
    # Обработка только указанного видео
    video_path = os.path.join(videos_dir, video_to_process)
    if os.path.isfile(video_path):
        plot_name = os.path.splitext(video_to_process)[0] + '_avg_med.png'
        plot_path = os.path.join(plots_dir, plot_name)
        elipsis_path = os.path.join(elipsis_dir, plot_name)
        analyze_path = os.path.join(analyze_dir, os.path.splitext(video_to_process)[0] + '.txt')
        process_video(video_path, plot_path, elipsis_path, analyze_path)
        print(f"Обработано видео: {video_to_process}, график сохранен в: {plot_path}")
    else:
        print(f"Видео {video_to_process} не найдено в {videos_dir}")
else:
    # Обработка всех видео из папки
    for video_name in os.listdir(videos_dir):
        video_path = os.path.join(videos_dir, video_name)
        plot_name = os.path.splitext(video_name)[0] + '.png'
        plot_path = os.path.join(plots_dir, plot_name)

        elipsis_path = os.path.join(elipsis_dir, plot_name)
        analyze_path = os.path.join(analyze_dir, os.path.splitext(video_name)[0] + '.txt')
        print(plot_path, elipsis_path, analyze_path)
        process_video(video_path, plot_path, elipsis_path, analyze_path)

        print(f"Обработано видео: {video_name}, график сохранен в: {plot_path}")