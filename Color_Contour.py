import cv2
import numpy as np


blueLower = np.array([120, 0, 0])    # 추출할 색의 하한(BGR)
blueUpper = np.array([250, 100, 10])    # 추출할 색의 상한(BGR)
yellowLower = np.array([0, 150, 190])    # 추출할 색의 하한(BGR)   // 0, 130 ,150
yellowUpper = np.array([100, 220, 250])    # 추출할 색의 상한(BGR)  // 55,250 ,250
BGRlower_Upper = [ 
    (blueLower,blueUpper),        # 0 : Blue
    (yellowLower,yellowUpper)     # 1 : Yellow 
                  ]               # 2 : Red

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
Ypointx = 0
Ypointy = 0

def to_Gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    return thresh

def color_select(index):
    mask = cv2.inRange(frame, BGRlower_Upper[index][0], BGRlower_Upper[index][1])  ## Color Mask
    Color = cv2.bitwise_and(frame, frame, mask=mask)             # 인덱스 색깔 정
    Color = to_Gray(Color)
    
    return Color
Color_name = ["BLUE","YELLOW"]
while cv2.waitKey(33)!=ord('q'):
    ret, frame = capture.read()
    height,width,_ = frame.shape
    Object = [[] for i in range(2)]
    Object_Contour = [[] for i in range(2)] 
    for i in range(2):
        Object[i] = color_select(i)
        Object_Contour[i], _ = cv2.findContours(Object[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in Object_Contour[i]:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(frame, [approx], -1, (0,255,0), 3)
            Point = np.mean(contour,axis=0)

            cv2.putText(frame,Color_name[i],np.int0(Point[0]),1,2,(0,255,0),2)
            cv2.circle(frame,np.int0(Point[0]),2,(0,255,255),-1)
            # cv2.drawContours(frame, [contour], -1, (0,0,255), 3)  

    cv2.imshow("VideoFrame", frame)
capture.release()
cv2.destroyAllWindows()