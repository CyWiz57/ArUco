import cv2
import cv2.aruco as aruco
import numpy as np
import os

import arucoModule as arm

url = 'https://10.244.121.165:8080/video'
cap = cv2.VideoCapture(url)
augDics = arm.loadAugImages("Resources")
while True:
    success, img = cap.read()
    arucoFound = arm.findArucoMarkers(img)
    # Loop through all the markers and augment each one
    if len(arucoFound[0]) != 0:
        for bbox, id in zip(arucoFound[0], arucoFound[1]):
            if int(id) in augDics.keys():
                img = arm.augmentAruco(bbox, id, img, augDics[int(id)])

    if img is not None:
        cv2.imshow('img', img)
    q = cv2.waitKey(1)
    if q == ord("q"):
        break
cv2.destroyAllWindows()



