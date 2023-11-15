import numpy as np 
import cv2 
import time
from realsense_depth import *
import multiprocessing





real_sqaure_size = 5 #실제 네모 큐브의 길이 (cm)
# Capturing video through webcam 
webcam = DepthCamera()

turtle_num =2 # 1번은 노랑, 2번은 파랑....
count = 10 # 한 번 계산에 turtle_point를 선택할 것인가
turtle_point = np.full((turtle_num, count), None, dtype=object)
m_turtle_point = np.empty((turtle_num, 2), dtype=int)
color_count = np.zeros (turtle_num, dtype=int) 

#fun_thing = np.zeros (turtle_num, dtype=int) 
sum_real_size_x = np.zeros((count)) #결과 값들을 모은 총결과들의 집합
sum_real_size_y = np.zeros((count)) #결과 값들을 모은 총결과들의 집합



re_turtle_point = np.empty((turtle_num,2,1) ,dtype=int)
real_size = np.empty((turtle_num,2) ,dtype=float) #실제 turtlebot 까지의 길이 (x, y)
re_left_side = np.array([[0, 0], [0, 0]])
x_vector = np.empty((0,2), dtype=int)
y_vector = np.empty((0,2), dtype=int)
pixel_x_size = 0
pixel_y_size = 0
zero = np.array([0,0]) #원점.

color = ["파랑", "노랑"]



def get_point(pic, contour, col):
    area = cv2.contourArea(contour) 
    if(area > 300): 
        #print('blue', pic)
        rect = cv2.minAreaRect(contour) 
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        ang = np.empty((0,2), dtype=int)
        for i in range(4):
            # box[i][0] = int64
            ang = np.append(ang, [[box[i][0], box[i][1]]], axis=0)
            #imageFrame = cv2.circle(imageFrame,(box[i][0], box[i][1]),10,(255,0,0),-1)
        D= ang.mean(axis=0) #원점.
        D = D.ravel()
        D = np.int32(D)
        # print(D)
        # print(zero)
        D[0] -= zero[0]
        D[1] -= zero[1]

        if(color_count[col]<count):
            turtle_point[col][color_count[col]] = D # 노랑색은 터틀 1번, 해당 터틀봇의 좌표를 turtle_point[0]에 넣음.
            color_count[col] += 1
            

        if(color_count[col]>=count):
            x_value = np.zeros(color_count[col])
            y_value = np.zeros(color_count[col])
            #print("blue")
            #print(turtle_point[0])
        
            for i in range(color_count[col]):
                x_value[i] = turtle_point[col][i][0]
                x_median = np.median(x_value)
                y_value[i] = turtle_point[col][i][1]
                y_median = np.median(y_value)

            get_y = int(x_median)
            get_x = int(y_median)
            m_turtle_point[col] = (get_x, get_y)
            
            

            #imageFrame = cv2.circle(imageFrame,(get_x, get_y),10,(0,255,255),-1)
            color_count[col] = 0
            turtle_point[col] = np.zeros(shape=turtle_point[col].shape)
        
            re_turtle_point[col] = np.array([[int(m_turtle_point[col][0])], [int(m_turtle_point[col][1])]] )
            re_turtle_point[col] = np.dot(re_left_side, re_turtle_point[col]) # 원점을 중심, 원점의 x, y 기저 벡터를 통한 새로운 좌표  
            
            real_size[col][0] = real_sqaure_size*re_turtle_point[col][0]/pixel_x_size
            real_size[col][1] = real_sqaure_size*re_turtle_point[col][1]/pixel_y_size  
            

            print(color[col], " 실제 좌표 " , real_size[col])
            #return real_size[col], col

# Start a while loop 


if __name__ == '__main__':

    pool = multiprocessing.Pool(processes=turtle_num)
    
    while(1): 
        
        # Reading the video from the 
        # webcam in image frames 
        _ ,_,imageFrame= webcam.get_frame()
        imageFrame = imageFrame[:,200:1000]
        
        # Convert the imageFrame in  
        # BGR(RGB color space) to  
        # HSV(hue-saturation-value) 
        # color space  
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 
    
        # Set range for red color and       
        # define mask 
        red_lower = np.array([170, 70, 110], np.uint8) 
        red_upper = np.array([180, 230, 255], np.uint8) 
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 
        
        
        purple_lower = np.array([160, 70, 110], np.uint8) 
        purple_upper = np.array([170, 230, 255], np.uint8) 
        purple_mask = cv2.inRange(hsvFrame, purple_lower, purple_upper)
        # Set range for green color and  
        # define mask 
        green_lower = np.array([25, 52, 72], np.uint8) 
        green_upper = np.array([102, 255, 255], np.uint8) 
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 
    
        # Set range for yellow color and 
        # define mask 
        yellow_lower = np.array([20, 10, 254], np.uint8)
        yellow_upper = np.array([30, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

        blue_lower = np.array([90, 50, 50], np.uint8)
        blue_upper = np.array([130, 255, 255], np.uint8)
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

        # Morphological Transform, Dilation 
        # for each color and bitwise_and operator 
        # between imageFrame and mask determines 
        # to detect only that particular color 
        kernel = np.ones((5, 5), "uint8") 
        
        # For red color 
        red_mask = cv2.dilate(red_mask, kernel) 
        res_red = cv2.bitwise_and(imageFrame, imageFrame,  
                                mask = red_mask) 
        
        # For green color 
        green_mask = cv2.dilate(green_mask, kernel) 
        res_green = cv2.bitwise_and(imageFrame, imageFrame, 
                                    mask = green_mask) 
        
        # For yellow color 
        yellow_mask = cv2.dilate(yellow_mask, kernel) 
        res_yellow = cv2.bitwise_and(imageFrame, imageFrame, 
                                mask = yellow_mask) 
    
        # For yellow color 
        blue_mask= cv2.dilate(blue_mask, kernel) 
        res_blue = cv2.bitwise_and(imageFrame, imageFrame, 
                                mask = blue_mask) 
        # For purple color 
        purple_mask = cv2.dilate(purple_mask, kernel) 
        res_purple = cv2.bitwise_and(imageFrame, imageFrame, 
                                mask = purple_mask) 
    
        

        # Creating contour to track red color 
        contours, hierarchy = cv2.findContours(red_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE) 
        
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 300): 
                rect = cv2.minAreaRect(contour) 
                box = cv2.boxPoints(rect)
                box = np.int0(box) #박스에는 x(4, 2) (x, y) 이렇게 구성되어있다.          
                
                ang = np.empty((0,2), dtype=int)
                for i in range(4):
                    # box[i][0] = int64
                    ang = np.append(ang, [[box[i][0], box[i][1]]], axis=0)
                    imageFrame = cv2.circle(imageFrame,(box[i][0], box[i][1]),10,(0,0,255),-1) 

                sums = np.sum(ang, axis=1)
                min = np.argmin(sums)
                diff = np.diff(ang, axis=1)
                top_left = ang[min]
                top_right = ang[np.argmin(diff)]
                bottom_left = ang[np.argmax(diff)]
                
            
                
                
                pixel_x_size = np.sqrt((top_left[0]-top_right[0])**2 + (top_left[1]-top_right[1])**2 )
                pixel_y_size = np.sqrt((top_left[0]-bottom_left[0])**2 + (top_left[1]-bottom_left[1])**2)
                x_vector = ((top_right[0]-top_left[0])/pixel_x_size, (top_right[1]-top_left[1])/pixel_x_size) 
                y_vector = ((top_left[0]-bottom_left[0])/pixel_y_size, (top_left[1]-bottom_left[1])/pixel_y_size)
                
                
                #if ( pixel_x_size == pixel_y_size) :
                # print("red x 픽셀 크기 : ", pixel_x_size)
                # print("red y 픽셀 크기 : ", pixel_y_size)
                # print("x 벡터 : ", x_vector)
                # print("y 벡터 : ", y_vector)

                left_side = np.array([[x_vector[0], y_vector[0]], [x_vector[1], y_vector[1]]])
                re_left_side = np.linalg.inv(left_side)

                #print(x_vector)
                #print(y_vector)
                #print(top_left, top_right, bottom_left)
                D= ang.mean(axis=0) #원점.
                D = D.ravel()
                D = np.int32(D)
                zero = D
                #print(D.shape)
                cv2.circle(imageFrame,zero,10,(0,0,255),-1)
                #print("red=", zero)
                # print("ang.shape = ", ang.shape) 
            # else :

                #   continue
        
        
        # Creating contour to track green color 
                
        
        
        contours, hierarchy = cv2.findContours(blue_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE) 
        
        for pic, contour in enumerate(contours): 
            pool.starmap(get_point, [pic, contour, 0])
            #get_point(pic, contour, 0)
            #print(color[0], real_size[0])
            #output_list = pool.starmap(get_point, [pic, contour, 0])       
                
                

    
        contours, hierarchy = cv2.findContours(yellow_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE) 
        
        for pic, contour in enumerate(contours): 
            pool.starmap(get_point, [pic, contour, 1])
            #get_point(pic, contour, 1)
            #print(color[1], real_size[1])
            # output_list = pool.starmap(get_point, [pic, contour, 1])
            
                #print(real_size)
        pool.close()
        pool.join()

        # Program Termination 
        
        cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame) 
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            webcam.release() 
            cv2.destroyAllWindows() 
            break