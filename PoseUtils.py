import cv2 as cv
import numpy as np


def get_rgb_from_bgr(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def get_bgr_from_rgb(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle