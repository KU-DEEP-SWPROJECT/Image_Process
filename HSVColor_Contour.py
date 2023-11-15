import cv2
import numpy as np
import Color_Object as ob
from realsense_depth import *
import pyrealsense2
import time
from Map import *

lower_yellow = (20, 10, 254)
upper_yellow = (30, 240, 255)

lower_green = (50, 30, 100)
upper_green = (80, 50, 150)        # NO

lower_red = (170,70,110)
upper_red = (180,230,255)

lower_purple = (160,70,110)
upper_purple = (170,230,255)

lower_blue = (38, 0, 255)        # NO
upper_blue = (38, 60, 255)


HSVlower = [lower_yellow,lower_blue,lower_green,lower_red,lower_purple]
HSVUpper = [upper_yellow,upper_blue,upper_green,upper_red,upper_purple]
Color_name = ["yellow","Blue","Green","Red"]
Object = [ob.Color_Object(color) for color in Color_name]

dc = DepthCamera()

def to_Gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Gray",gray)
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    return thresh

def color_select(index):
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img,HSVlower[index],HSVUpper[index])
    Color = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("BITAND",Color)
    Color = to_Gray(Color)            # to Gray because threshold
    
    return Color

while cv2.waitKey(33)!=ord('q'):
    print(robots)

    ret,_, frame = dc.get_frame()
    for i in range(len(Color_name)):
        POINTS = color_select(i)
        contours, _ = cv2.findContours(POINTS, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area>200 and area < 300:
                print(Color_name[i])
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                center = np.mean(contour,axis=0)
                cv2.drawContours(frame, [approx], -1, (0,255,0), 3)
                input_robots(i+1,np.int32(center).ravel())
    cv2.imshow("VideoFrame", frame)

cv2.destroyAllWindows()