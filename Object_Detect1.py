import cv2
import numpy as np
import Color_Object as ob
from realsense_depth import *
import pyrealsense2


dc = DepthCamera()
object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=20)


while cv2.waitKey(33)!=ord('q'):
    ret,_, frame = dc.get_frame() 
    roi = frame[:][100:500]
    mask = object_detector.apply(roi)

    contours , _ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            cv2.drawContours(roi,[contour],-1,(0,255,0),2)
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("mask",mask)
    cv2.imshow("VideoFrame", frame)
cv2.destroyAllWindows()