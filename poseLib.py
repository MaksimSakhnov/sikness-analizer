import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

    tmp = module(x1, y1) * module(x2, y2)
    if tmp == 0:
        return 0
    cos = scalar(x1, y1, x2, y2) / tmp
    if -1 <= cos <= 1:
        return math.degrees(math.acos(cos))
    return 0


def draw_connections(frame, keypoints, edges, confidence_threshold, edges_vec, edges_deg, result, result_edges):
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
