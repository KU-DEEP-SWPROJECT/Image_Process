import cv2
import numpy as np
import Color_Object as ob
from realsense_depth import *

BGRlower_Upper = [ 
    (ob.blueLower,ob.blueUpper),        # 0 : Blue
    (ob.yellowLower,ob.yellowUpper)    # 1 : Yellow 
                  ]                     # 2 : Black
Color_name = ["BLUE","YELLOW","Black"]
Object = [ob.Color_Object(Color_name[i]) for i in range(2)]
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

def to_Gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    return thresh

def color_select(index):
    mask = cv2.inRange(frame, BGRlower_Upper[index][0], BGRlower_Upper[index][1])  ## Color Mask
    Color = cv2.bitwise_and(frame, frame, mask=mask)      # Index Color bit and operation
    Color = to_Gray(Color)                                # to Gray because threshold
    
    return Color

dc = DepthCamera()
while cv2.waitKey(33)!=ord('q'):
    ret, _,frame = dc.get_frame()
    frame = frame[:340,100:400]
    for i in range(2):
        Object[i].get_points(color_select(i))
        contours, _ = cv2.findContours(Object[i].points, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        corners = cv2.goodFeaturesToTrack(Object[i].points,4,0.5,50)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                epsilon = 0.02 * cv2.arcLength(np.int32(contour), True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(frame, [approx], -1, (0,255,0), 3)
                center = np.mean(contour,axis=0)

                cv2.putText(frame,Color_name[i],np.int32(center[0]),1,2,(0,255,0),2)
                cv2.circle(frame,np.int32(center[0]),1,(0,255,255),-1)
                cv2.drawContours(frame, [contour], -1, (0,0,255), 2) 
                print(center)
     

    cv2.imshow("VideoFrame", frame)
dc.release()
cv2.destroyAllWindows()