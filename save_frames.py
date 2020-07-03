import os
import cv2
import numpy as np


cap = cv2.VideoCapture("clip.mp4")


def get_frame(sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = cap.read()
    if hasFrames:
        cv2.imwrite("images_clip/image" + str(count) + ".jpg", image)
    return hasFrames


sec = 0
frame_rate = 0.1
count = 1
success = get_frame(sec)
while success:
    count = count + 1
    sec = sec + frame_rate
    sec = round(sec, 2)
    success = get_frame(sec)
