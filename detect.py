import cv2
import numpy as np

capture = cv2.VideoCapture(0)

class_file = 'class.names'
class_names = []

with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

while True:
    success, img = capture.read()

    cv2.imshow('Image', img)
    cv2.waitKey(1)