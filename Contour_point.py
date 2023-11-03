import cv2
import numpy as np
import Color_Object as ob


BGRlower_Upper = [ 
    (ob.blueLower,ob.blueUpper),        # 0 : Blue
    (ob.yellowLower,ob.yellowUpper),
    (ob.blackLower,ob.blackUpper)     # 1 : Yellow 
                  ]                     # 2 : Black
Color_name = ["BLUE","YELLOW","Black"]
Object = [ob.Color_Object(Color_name[i]) for i in range(3)]
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

def to_Gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    return thresh

def color_select(index):
    mask = cv2.inRange(frame, BGRlower_Upper[index][0], BGRlower_Upper[index][1])  ## Color Mask
    Color = cv2.bitwise_and(frame, frame, mask=mask)      # Index Color bit and operation
    Color = to_Gray(Color)                                # to Gray because threshold
    
    return Color


while cv2.waitKey(33)!=ord('q'):
    ret, frame = capture.read() 
    height,width,_ = frame.shape  # 영상 세로,가로
    for i in range(3):
        Object[i].get_points(color_select(i))
        contours, _ = cv2.findContours(Object[i].points, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        corners = cv2.goodFeaturesToTrack(Object[i].points,4,0.5,50)
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(frame, [approx], -1, (0,255,0), 3)
            center = np.mean(contour,axis=0)

            cv2.putText(frame,Color_name[i],np.int32(center[0]),1,2,(0,255,0),2)
            cv2.circle(frame,np.int32(center[0]),2,(0,255,255),-1)
            # cv2.drawContours(frame, [contour], -1, (0,0,255), 3) 
        if corners is not None and corners.all():
            for corner in corners:
                x,y = corner.ravel()
                cv2.circle(frame,(int(x),int(y)),10,(255,0,0),-1)    


    cv2.imshow("VideoFrame", frame)
capture.release()
cv2.destroyAllWindows()