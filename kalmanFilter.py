import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter

# =============================
# 1. Kalman Filter 2D
# =============================
class KalmanFilter2D:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 0.033
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.P *= 1000.
        self.kf.Q = np.eye(4) * 0.01
        self.kf.R = np.eye(2) * 5
        self.initialized = False

    def update(self, measurement):
        if measurement is None:
            self.kf.predict()
        else:
            if not self.initialized:
                x, y = measurement
                self.kf.x = np.array([[x], [y], [0], [0]])
                self.initialized = True
            else:
                self.kf.predict()
                self.kf.update(np.array([[measurement[0]], [measurement[1]]]))
        return float(self.kf.x[0]), float(self.kf.x[1])

# =============================
# 2. Moving Average Filter
# =============================
class MovingAverageFilter2D:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer_x = deque(maxlen=window_size)
        self.buffer_y = deque(maxlen=window_size)

    def update(self, measurement):
        if measurement is not None:
            self.buffer_x.append(measurement[0])
            self.buffer_y.append(measurement[1])
        if len(self.buffer_x) == 0:
            return None
        avg_x = np.mean(self.buffer_x)
        avg_y = np.mean(self.buffer_y)
        return avg_x, avg_y

# =============================
# 3. Exponential Smoothing Filter
# =============================
class ExponentialSmoothingFilter2D:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.prev = None

    def update(self, measurement):
        if measurement is None:
            return self.prev
        if self.prev is None:
            self.prev = measurement
        else:
            x = self.alpha * measurement[0] + (1 - self.alpha) * self.prev[0]
            y = self.alpha * measurement[1] + (1 - self.alpha) * self.prev[1]
            self.prev = (x, y)
        return self.prev

# =============================
# 4. Median Filter
# =============================
class MedianFilter2D:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer_x = deque(maxlen=window_size)
        self.buffer_y = deque(maxlen=window_size)

    def update(self, measurement):
        if measurement is not None:
            self.buffer_x.append(measurement[0])
            self.buffer_y.append(measurement[1])
        if len(self.buffer_x) == 0:
            return None
        med_x = np.median(self.buffer_x)
        med_y = np.median(self.buffer_y)
        return med_x, med_y


class CombinedMedianKalman:
    def __init__(self, window_size=5):
        self.median_filter = MedianFilter2D(window_size)
        self.kalman = KalmanFilter2D()

    def update(self, value):
        filtered = self.median_filter.update(value)
        if filtered is None:
            # Если медианный фильтр вернул None, передаем None в Калман
            return self.kalman.update(None)
        else:
            # Иначе передаем фильтрованное значение Калману
            return self.kalman.update(filtered)


class CombinedExponentialKalman:
    def __init__(self, alpha=0.3):
        self.exp_filter = ExponentialSmoothingFilter2D(alpha)
        self.kalman = KalmanFilter2D()

    def update(self, value):
        filtered = self.exp_filter.update(value)
        if filtered is None:
            return self.kalman.update(None)
        else:
            return self.kalman.update(filtered)


class CombinedMovingAverageKalman:
    def __init__(self, window_size=5):
        self.moving_avg = MovingAverageFilter2D(window_size)
        self.kalman = KalmanFilter2D()

    def update(self, value):
        filtered = self.moving_avg.update(value)
        if filtered is None:
            return self.kalman.update(None)
        else:
            return self.kalman.update(filtered)


class CombinedMedianMovingAverage:
    def __init__(self, median_window=5, moving_window=5):
        self.median_filter = MedianFilter2D(median_window)
        self.moving_avg = MovingAverageFilter2D(moving_window)

    def update(self, value):
        med_filtered = self.median_filter.update(value)
        if med_filtered is None:
            return None
        final_filtered = self.moving_avg.update(med_filtered)
        return final_filtered