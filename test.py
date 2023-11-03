import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)


while cv2.waitKey(33)!=ord('q'):
    ret, frame = capture.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 120, 255, 1)
    corners = cv2.goodFeaturesToTrack(canny,4,0.5,50)
    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(frame,(int(x),int(y)),10,(255,0,0),-1)    

    cv2.imshow("canny",canny)
    cv2.imshow("VideoFrame", frame)
capture.release()
cv2.destroyAllWindows()