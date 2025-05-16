import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import ast
from sklearn.cluster import DBSCAN
from matplotlib.patches import Ellipse
from scipy.fftpack import fft
from sklearn.preprocessing import MinMaxScaler

KEYPOINT_DICT = {
    0: 'нос',
    1: 'левый глаз',
    2: 'правый глаз',
    3: 'левое ухо',
    4: 'правое ухо',
    5: 'левое плечо',
    6: 'правое плечо',
    7: 'левый локоть',
    8: 'правый локоть',
    9: 'левое запястье',
    10: 'правое запястье',
    11: 'левое бедро',
    12: 'правое бедро',
    13: 'левое колено',
    14: 'правое колено',
    15: 'левая лодыжка',
    16: 'правая лодыжка'
}

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


def calculate_angle_for_midline(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return math.degrees(math.atan2(dy, dx))


def calculate_mid_angle(mid_line):
    angles = []
    deviations = []
    displacements = []

    x1_first, y1_first = mid_line[0][0], mid_line[1][0]
    x2_first, y2_first = mid_line[0][1], mid_line[1][1]

    x_mid_first = (x1_first + x2_first) / 2
    y_mid_first = (y1_first + y2_first) / 2

    theta_first = calculate_angle_for_midline(x1_first, y1_first, x2_first, y2_first)
    print(f"Угол первой прямой: {theta_first:.2f} градусов")
    print(f"Средняя точка первой прямой: ({x_mid_first:.2f}, {y_mid_first:.2f})")

    step = 2
    for i in range(0, len(mid_line[0]) - 1, step):
        x1, y1 = mid_line[0][i], mid_line[1][i]
        x2, y2 = mid_line[0][i + 1], mid_line[1][i + 1]

        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2

        delta_x = x_mid - x_mid_first
        displacements.append(delta_x)

        theta_new = calculate_angle_for_midline(x1, y1, x2, y2)

        deviation = theta_new - theta_first
        deviations.append(deviation)

        print(f"Отрезок {i}: ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})")
        print(f"Средняя точка: ({x_mid:.2f}, {y_mid:.2f})")
        print(f"Угол: {theta_new:.2f} градусов, Отклонение: {deviation:.2f} градусов")
        print(f"Смещение по X: {delta_x:.2f} пикселей")

        angles.append(theta_new)

    return deviations, displacements


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = keypoints

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_center_of_mass(frame, keypoints, confidence_threshold=0.4, kf_filter=None, center_coords=None):
    y, x, c = frame.shape
    shaped = keypoints

    def weighted_center_of_mass_local(keypoints_local, conf_thresh):
        segments = {
            'torso': {'points': [11, 12], 'weight': 0.50},
            'thigh_l': {'points': [11, 13], 'weight': 0.10},
            'thigh_r': {'points': [12, 14], 'weight': 0.10},
            'calf_l': {'points': [13, 15], 'weight': 0.05},
            'calf_r': {'points': [14, 16], 'weight': 0.05},
            'head': {'points': [0], 'weight': 0.08},
            'shoulders': {'points': [5, 6], 'weight': 0.12},
        }

        total_weight = 0
        sum_x = 0
        sum_y = 0

        for seg in segments.values():
            pts = seg['points']
            w = seg['weight']
            coords = []

            valid = True
            for i in pts:
                y_kp, x_kp, conf = keypoints_local[i]
                if conf < conf_thresh:
                    valid = False
                    break
                coords.append((x_kp, y_kp))

            if valid:
                avg_x = np.mean([pt[0] for pt in coords])
                avg_y = np.mean([pt[1] for pt in coords])
                sum_x += avg_x * w
                sum_y += avg_y * w
                total_weight += w

        if total_weight > 0:
            center_x = sum_x / total_weight
            center_y = sum_y / total_weight
            return (center_x, center_y)
        else:
            return None

    center_raw = weighted_center_of_mass_local(shaped, confidence_threshold)

    if kf_filter is not None:
        center = kf_filter.update(center_raw)
    else:
        center = center_raw

    if center:
        cv2.circle(frame, (int(center[0]), int(center[1])), 6, (255, 255, 0), -1)

        if center and all(np.isfinite(center)) and center_coords is not None:
            center_coords.append([float(center[0]), float(center[1])])


""" Возвращает угол между двумя векторами """


def angle_between(x1, y1, x2, y2):
    def scalar(x1, y1, x2, y2):
        return x1 * x2 + y1 * y2

    def module(x, y):
        return math.sqrt(x ** 2 + y ** 2)

    tmp = module(x1, y1) * module(x2, y2)
    if tmp == 0:
        return 0
    cos = scalar(x1, y1, x2, y2) / tmp
    if -1 <= cos <= 1:
        return math.degrees(math.acos(cos))
    return 0




def compute_metrics(points, fps=30):
    points = np.array(points)
    points = points[np.all(np.isfinite(points), axis=1)]
    """ Рассчитывает метрики устойчивости """
    t = np.arange(len(points)) / fps  # Временные метки
    x, y = points[:, 0], points[:, 1]

    # 1. Амплитуда колебаний
    std_x, std_y = np.std(x), np.std(y)
    range_x, range_y = np.ptp(x), np.ptp(y)  # max - min

    # 2. RMS отклонений
    rms_x, rms_y = np.sqrt(np.mean((x - np.mean(x))**2)), np.sqrt(np.mean((y - np.mean(y))**2))

    # 3. Скорость и ускорение
    dt = 1 / fps
    dx, dy = np.diff(x) / dt, np.diff(y) / dt  # Скорость
    ddx, ddy = np.diff(dx) / dt, np.diff(dy) / dt  # Ускорение

    # 4. Длина пути
    path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

    # 5. Среднее отклонение (направление наклона)
    mean_x_offset = np.mean(x - np.mean(x))  # > 0 — наклон вправо, < 0 — влево

    # 6. Частотный анализ (FFT)
    fft_x = np.abs(fft(x - np.mean(x)))[:len(x) // 2]
    fft_y = np.abs(fft(y - np.mean(y)))[:len(y) // 2]
    dominant_freq_x = np.argmax(fft_x) / len(x) * fps
    dominant_freq_y = np.argmax(fft_y) / len(y) * fps

    return {
        'std_x': std_x, 'std_y': std_y,
        'range_x': range_x, 'range_y': range_y,
        'rms_x': rms_x, 'rms_y': rms_y,
        'mean_speed_x': np.mean(np.abs(dx)), 'mean_speed_y': np.mean(np.abs(dy)),
        'mean_accel_x': np.mean(np.abs(ddx)), 'mean_accel_y': np.mean(np.abs(ddy)),
        'path_length': path_length,
        'mean_x_offset': mean_x_offset,
        'dominant_freq_x': dominant_freq_x, 'dominant_freq_y': dominant_freq_y
    }



# Средние stability_score по классам из твоих данных
ref_class_means = {
    0: 13.48,
    1: 40.42,
    2: 59.17
}

def compute_stability_score(metrics):
    ref = {
        'std_x': (0.4116, 1.9780),
        'std_y': (0.2835, 5.2061),
        'range_x': (2.0694, 9.1295),
        'range_y': (1.7273, 22.2045),
        'rms_x': (0.4116, 1.9780),
        'rms_y': (0.2835, 5.2061),
        'mean_speed_x': (0.3276, 1.3184),
        'mean_speed_y': (0.3128, 12.1922),
        'mean_accel_x': (1.5570, 30.8647),
        'mean_accel_y': (1.9196, 302.7105),
        'path_length': (28.7214, 531.0236),
        'dominant_freq_x': (0.0202, 0.1868),
        'dominant_freq_y': (0.0233, 0.2088),
        'mean_x_offset': (0, 0),
    }

    neg_metrics = ['std_x', 'std_y', 'range_x', 'range_y', 'rms_x', 'rms_y']
    pos_metrics = ['mean_speed_x', 'mean_speed_y', 'mean_accel_x', 'mean_accel_y', 'path_length', 'dominant_freq_x', 'dominant_freq_y']

    scores = []

    for metric in neg_metrics + pos_metrics:
        val = metrics.get(metric, None)
        if val is None:
            continue
        min_val, max_val = ref[metric]
        if max_val == min_val:
            scaled = 0.5
        else:
            scaled = (val - min_val) / (max_val - min_val)
            if metric in neg_metrics:
                scaled = 1 - scaled
            scaled = max(0, min(1, scaled))
        scores.append(scaled)

    raw_score = np.mean(scores) * 100

    # Выбираем класс по исходному score с порогами (можно менять)
    if raw_score < 20:
        predicted_class = 0
    elif raw_score < 55:
        predicted_class = 1
    else:
        predicted_class = 2

    # Корректируем score ближе к среднему по классу (например, среднее с весом 0.7)
    stability_score = round(0.7 * raw_score + 0.3 * ref_class_means[predicted_class], 2)

    return stability_score, predicted_class


def plot_trajectory(points):
    points = np.array(points)
    points = points[np.all(np.isfinite(points), axis=1)]
    """ Строит график траектории + эллипс доверия """
    x, y = points[:, 0], points[:, 1]

    # Оценка эллипса доверия
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.arctan2(*eigenvectors[:, 0][::-1])
    width, height = 2 * np.sqrt(eigenvalues)  # 95% доверительный эллипс

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker='o', linestyle='-', alpha=0.6)

    # Добавляем эллипс доверия
    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)), width=width, height=height,
                      angle=np.degrees(angle), edgecolor='r', facecolor='none', linewidth=2)
    plt.gca().add_patch(ellipse)

    plt.xlabel("X (влево-вправо)")
    plt.ylabel("Y (вверх-вниз)")
    plt.title("Траектория движения центра тела")
    plt.grid()
    return plt


def draw_connections(frame, keypoints, edges, confidence_threshold, edges_vec, edges_deg, result, result_edges):
    y, x, c = frame.shape
    shaped = keypoints

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
            print("edge:", KEYPOINT_DICT[p1], " - ", KEYPOINT_DICT[p2], ",", "coords: (", int(x1), ",", int(y1), ";",
                  int(x2), ",", int(y2), ")", "deg:", deg)
            result[p1][0].append(x1)
            result[p2][0].append(x2)
            result[p1][1].append(y1)
            result[p2][1].append(y2)
            result_edges[edge][0].append(x1)
            result_edges[edge][0].append(x2)
            result_edges[edge][1].append(y1)
            result_edges[edge][1].append(y2)
            # обновляем словарь векторов
            edges_vec[color] = new_vec
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


def Average(lst):
    return sum(lst) / len(lst)


def find_midpoint(x1, y1, x2, y2):
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    return mid_x, mid_y


def merge_arrays(arr1, arr2):
    min_len = min(len(arr1[0]), len(arr2[0]))
    result = [[], []]
    for i in range(min_len):
        for j in range(2):
            if (j == 0):
                result[0].append(arr1[0][i])
                result[1].append(arr1[1][i])
            else:
                result[0].append(arr2[0][i])
                result[1].append(arr2[1][i])
    return result


def get_pose_line_coords(result_edges):
    shoulder_line = [[], []]
    hip_line = [[], []]
    for edge, data in result_edges.items():
        if (edge == (5, 6) or edge == (11, 12)):
            x, y = data
            len_x = len(x)
            if len_x % 2 == 0:
                len_x -= 1
            for i in range(0, len_x, 2):
                x1, y1 = x[i], y[i]
                x2, y2 = x[i + 1], y[i + 1]
                mid_x, mid_y = find_midpoint(x1, y1, x2, y2)
                if (edge == (5, 6)):
                    shoulder_line[0].append(mid_x)
                    shoulder_line[1].append(mid_y)
                else:
                    hip_line[0].append(mid_x)
                    hip_line[1].append(mid_y)
    return merge_arrays(shoulder_line, hip_line)


def get_diagnosis(max_angle, avg, left_s, right_s):
    dif = left_s - right_s
    abs_dif = abs(dif)
    text = ''
    asymmetry_text = ''
    left_right_text = 'влево'
    if dif < 0:
        left_right_text = 'вправо'
    if abs_dif < 4:
        text = 'нарушения равновесия нет'
        asymmetry_text = 'асимметрии нет'
        left_right_text = ''
    elif abs_dif < 10:
        text = 'незначительные нарушения равновесия'
        asymmetry_text = 'незначительная асимметрия'
    else:
        text = 'сильные нарушения равновесия'
        asymmetry_text = 'умеренная асимметрия'
    return text + ', ' + asymmetry_text + ' ' + left_right_text



def draw_plots(result_edges):
    # Создание второго графика
    for edge, data in result_edges.items():
        x, y = data
        if len(x) == 0 or len(y) == 0:
            continue
        p1, p2 = edge
        first_x = x[0]
        first_y = y[0]
        angles = []
        for i in range(len(x)):
            angles.append(angle_between(first_x, first_y, x[i], y[i]))
        # print(KEYPOINT_DICT[p1] + " - " + KEYPOINT_DICT[p2] + ": ", "min: " + str(min(angles) * 100 // 100),
        #       ", max: " + str(max(angles)))
        plt.plot(x, y, label=(KEYPOINT_DICT[p1] + " - " + KEYPOINT_DICT[p2] + " max: " + str(
            round(max(angles), 2)) + ", avg: " + str(round(Average(angles), 2))))

    mid_line = get_pose_line_coords(result_edges)
    print(mid_line)
    mid_angles, displacements = calculate_mid_angle(mid_line)
    max_angle = round(max(mid_angles), 2)
    avg = round(Average(mid_angles), 2)
    left_s = round(max(displacements), 2)
    right_s = abs(round(min(displacements), 2))
    plt.plot(mid_line[0], mid_line[1],
             label='MID (max angle: ' + str(max_angle) + ", avg: " + str(avg) + ', смещение влево px: '
                   + str(left_s) + ', смещение вправо px: ' + str(right_s) + ')')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.suptitle(get_diagnosis(max_angle, avg, left_s, right_s))
    plt.title("Движение точек (График 2)")
    plt.legend()

    # dict_with_str_keys = {str(k): v for k, v in result_edges.items()}
    #
    # with open('dict.json', 'w') as json_file:
    #     json.dump(dict_with_str_keys, json_file)

    # plt.tight_layout()  # Для автоматического выравнивания графиков
    plt.gca().invert_yaxis()
    return plt
