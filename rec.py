# 해당 코드는 원점을 찾아서, 원점의 (x,y) 좌표와 x, y벡터를 (a, b) 형태로 반환합니다.


import cv2
import numpy as np
import circle


get_img = cv2.imread("checking3.png")
new_width = 800
new_height = 800
img = cv2.resize(get_img, (new_width, new_height))

# 폰트 및 텍스트 스타일 설정
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (0, 0, 0)  # 텍스트의 색상 (흰색)
font_thickness = 1

# bgrUpper = (66, 246, 245)  #노랑색 큐브
# bgrUpper = (123, 149, 153) #밑 판의 색

bgrLower = (0, 100, 175)    # 추출할 색의 하한(BGR)
bgrUpper = (5, 176, 253)    # 추출할 색의 상한(BGR)
img_mask = cv2.inRange(img, bgrLower, bgrUpper) # BGR로 부터 마스크를 작성
res = cv2.bitwise_and(img, img, mask=img_mask)
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
contours, hierachy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#f = open("rect.txt",'w')c
data = contours
#print(data)



real_sqaure_size = 5 #실제 네모 큐브의 길이 (cm)



for contour in contours:
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(img, [approx], -1, (0,255,0), 3)
    
   # print(len(approx)) # 각 뭉터기가 가진 꼭짓점의 개수.
    po, row, col = approx.shape
    #print(approx.shape)
    if ( len(approx) > 6): # 원인 경우
        print("circle")
        # for point in approx:
            
    else :
        ang = np.empty((0,2), dtype=int)
        for point in approx: # 각 포인트에는 꼭짓점들이 존재.
            x = int (point[:,0]) 
            y = int (point[:,1])

            ang = np.append(ang, [[x, y]], axis=0)
            text = str(point)
            cv2.putText(img, text, ( x, y), font, font_scale, font_color, font_thickness)
            cv2.circle(img,(x, y),10,(255,0,0),-1) 
        square_num, each_ang_num = ang.shape
        #print(ang)
        
        sums = np.sum(ang, axis=1)
        min = np.argmin(sums)
        diff = np.diff(ang, axis=1)
        top_left = ang[min]
        top_right = ang[np.argmin(diff)]
        bottom_left = ang[np.argmax(diff)]
        
        x_vector = np.empty((0,2), dtype=int)
        y_vector = np.empty((0,2), dtype=int)
        
        
        pixel_x_size = np.sqrt((top_left[0]-top_right[0])**2 + (top_left[1]-top_right[1])**2 )
        pixel_y_size = np.sqrt((top_left[0]-bottom_left[0])**2 + (top_left[1]-bottom_left[1])**2)
        x_vector = ((top_right[0]-top_left[0])/pixel_x_size, (top_right[1]-top_left[1])/pixel_x_size) 
        y_vector = ((top_left[0]-bottom_left[0])/pixel_y_size, (top_left[1]-bottom_left[1])/pixel_y_size)
        
        print("x 픽셀 크기 : ", pixel_x_size)
        print("y 픽셀 크기 : ", pixel_y_size)
        print("x 벡터 : ", x_vector)
        print("y 벡터 : ", y_vector)

        left_side = np.array([[x_vector[0], y_vector[0]], [x_vector[1], y_vector[1]]])
        re_left_side = np.linalg.inv(left_side)

        #print(x_vector)
        #print(y_vector)
        #print(top_left, top_right, bottom_left)
        D= ang.mean(axis=0) #원점.
        D = D.ravel()
        D = np.int32(D)
        
        #print(D.shape)
        cv2.circle(img,D,10,(255,0,0),-1)
        
        turtle_num =2
        turtle_point = np.empty((turtle_num,2) ,dtype=int)
        re_turtle_point = np.empty((turtle_num,2,1) ,dtype=int)

        real_size = np.empty((turtle_num,2) ,dtype=float) #실제 turtlebot 까지의 길이 (x, y)

        turtle_color = np.empty((turtle_num,2), dtype=int) # (bgrl, bgru)로 된 배열
        turtle_color = [
            [(200, 100, 0), (232, 162, 0)],
            [(150, 50, 150), (164, 73, 163)]
        ]

        for i in range(turtle_num):
            turtle_point[i] = circle.find_turtle(get_img, turtle_color[i][0],turtle_color[i][1]) #픽셀에서의 좌표
            cv2.circle(img,turtle_point[i],10,(255,0,0),-1) 

            turtle_point[i][0] -= D[0]
            turtle_point[i][1] -= D[1]
            re_turtle_point[i] = np.array([[int(turtle_point[i][0])], [int(turtle_point[i][1])]] )
            # print(re_turtle_point[i])
            # print(np.dot(re_left_side, re_turtle_point[i]).shape)
            re_turtle_point[i] = np.dot(re_left_side, re_turtle_point[i]) # 원점을 중심, 원점의 x, y 기저 벡터를 통한 새로운 좌표  
            # print(re_turtle_point[i])
            #print("변환 = ",re_turtle_point[i])
            
            real_size[i][0] = real_sqaure_size*re_turtle_point[i][0]/pixel_x_size
            real_size[i][1] = real_sqaure_size*re_turtle_point[i][1]/pixel_y_size  
            
            print("실제 =", real_size[i])
        

        
    
    
cv2.imshow("Img",img)
cv2.waitKey(0)
#f.close()