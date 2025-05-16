from filterpy.kalman import KalmanFilter
import numpy as np

class KalmanFilter2D:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        dt = 0.033  # время между измерениями (кадрами)
        # Матрица перехода состояния (с учётом скорости)
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        # Матрица измерений (мы измеряем только позиции)
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])

        # Начальная ковариация ошибки (небольшая неуверенность)
        self.kf.P *= 1000.

        # Процессный шум (шум движения)
        self.kf.Q = np.eye(4) * 0.01

        # Измерительный шум (шум сенсора)
        self.kf.R = np.eye(2) * 5

        # Начальное состояние (x, y, vx, vy)
        self.initialized = False

    def update(self, measurement):
        if measurement is None:
            # Нет измерения — просто прогнозируем дальше
            self.kf.predict()
        else:
            if not self.initialized:
                # Инициализация состояния по первому измерению
                x, y = measurement
                self.kf.x = np.array([[x], [y], [0], [0]])
                self.initialized = True
            else:
                self.kf.predict()
                self.kf.update(np.array([[measurement[0]], [measurement[1]]]))

        # Возвращаем текущую оценку позиции
        return float(self.kf.x[0]), float(self.kf.x[1])
