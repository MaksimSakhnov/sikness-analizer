import cv2
import numpy as np  # для работы с математикой
import matplotlib.pyplot as plt  # для вывода картинки

def subtract_images():
    circle = cv2.imread('spliced_video/bottom_left_0.jpg')
    star = cv2.imread('spliced_video/bottom_left_1.jpg')
    subtracted = cv2.subtract(star, circle)
    plt.imshow(subtracted)
    plt.show()


subtract_images()