import cv2
import numpy as np
import Color_Object as ob
from realsense_depth import *
import pyrealsense2

yellowLower = (30,  100, 10)
yellowUpper = (70, 255, 255)
orangeLower = (100, 200, 200)
orangeUpper = (140, 255, 255)
HSVlower_Upper = [ 
    (yellowLower,yellowUpper),
    
    
                  ]                     # 2 : Black
Color_name = ["Yellow","Orange"]
Object = [ob.Color_Object(Color_name[i]) for i in range(2)]

dc = DepthCamera()

def to_Gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Gray",gray)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    return thresh

def color_select(index):
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img,HSVlower_Upper[index][0],HSVlower_Upper[index][1])
    Color = cv2.bitwise_and(frame, frame, mask=mask)      # Index Color bit and operation
    cv2.imshow("bITAND",Color)
    Color = to_Gray(Color)                                # to Gray because threshold
    
    return Color


while cv2.waitKey(33)!=ord('q'):
    ret,_, frame = dc.get_frame() 
    height,width,_ = frame.shape  # 영상 세로,가로
    for i in range(2):
        Object[i].get_points(color_select(i))
        contours, _ = cv2.findContours(Object[i].points, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(frame, [approx], -1, (0,255,0), 3)
            center = np.mean(contour,axis=0)

            # cv2.putText(frame,Color_name[i],np.int32(center[0]),1,2,(0,255,0),2)
            cv2.circle(frame,np.int32(center[0]),2,(0,255,255),-1)
            # cv2.drawContours(frame, [contour], -1, (0,0,255), 3) 

    cv2.imshow("VideoFrame", frame)
cv2.destroyAllWindows()