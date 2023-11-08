# 해당 코드는 원점을 찾아서, 원점의 (x,y) 좌표와 x, y벡터를 (a, b) 형태로 반환합니다.


import cv2
import numpy as np

def find_turtle(image, bgrl, bgru):
    get_img = image
    new_width = 800
    new_height = 800
    img = cv2.resize(get_img, (new_width, new_height))


    bgrLower = bgrl    # 추출할 색의 하한(BGR)
    bgrUpper = bgru    # 추출할 색의 상한(BGR)


    img_mask = cv2.inRange(img, bgrLower, bgrUpper) # BGR로 부터 마스크를 작성
    res = cv2.bitwise_and(img, img, mask=img_mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, hierachy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(img, [approx], -1, (0,255,0), 3)
        
    # print(len(approx)) # 각 뭉터기가 가진 꼭짓점의 개수.
        po, row, col = approx.shape
        #print(approx.shape)
        if ( len(approx) > 6): # 원인 경우
            ang = np.empty((0,2), dtype=int)
            for point in approx: # 각 포인트에는 꼭짓점들이 존재.
                x = int (point[:,0]) 
                y = int (point[:,1])

                ang = np.append(ang, [[x, y]], axis=0)
            #print(ang)
            D= ang.mean(axis=0) #원점.
            D = D.ravel()
            D = np.int32(D)
            return D
            
                
        else :
            print("square")
            
            

        
   