# -*- coding: utf-8 -*-
import cv2
import os

fps = 20
image_folder = './Figures'
video_name = './Ani.avi'

images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
self_cap = cv2.VideoCapture(0)
self_fourcc = cv2.VideoWriter_fourcc(*'MP42')
video = cv2.VideoWriter(video_name, self_fourcc, fps, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release
