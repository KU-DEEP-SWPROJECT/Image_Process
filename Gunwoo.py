import numpy as np 
import cv2 
import time
from realsense_depth import *
  
real_sqaure_size = 4 #실제 네모 큐브의 길이 (cm)
# Capturing video through webcam 
webcam = DepthCamera()

turtle_num =1 # 1번은 노랑, 2번은 파랑....
count = 20 # 한 번 계산에 turtle_point를 선택할 것인가
turtle_point = np.full((turtle_num, count), None, dtype=object)
m_turtle_point = np.empty((turtle_num, 2), dtype=int)
color_count = np.zeros (turtle_num, dtype=int) 

zero = np.array([0,0]) #원점.
re_turtle_point = np.empty((turtle_num,2,1) ,dtype=int)
real_size = np.empty((turtle_num,2) ,dtype=float) #실제 turtlebot 까지의 길이 (x, y)
re_left_side = np.array([[0, 0], [0, 0]])
x_vector = np.empty((0,2), dtype=int)
y_vector = np.empty((0,2), dtype=int)
pixel_x_size = 0
pixel_y_size = 0

# Start a while loop 
while(1): 
    
    # Reading the video from the 
    # webcam in image frames 
    _, _,imageFrame= webcam.get_frame() 
    
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
  
    # Set range for green color and  
    # define mask 
    green_lower = np.array([25, 52, 72], np.uint8) 
    green_upper = np.array([102, 255, 255], np.uint8) 
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 
  
    # Set range for yellow color and 
    # define mask 
    yellow_lower = np.array([26, 87, 111], np.uint8)
    yellow_upper = np.array([38, 255, 255], np.uint8)
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
                imageFrame = cv2.circle(imageFrame,(box[i][0], box[i][1]),10,(255,0,0),-1) 

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
            
            # print("x 픽셀 크기 : ", pixel_x_size)
            # print("y 픽셀 크기 : ", pixel_y_size)
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
            cv2.circle(imageFrame,zero,10,(255,0,0),-1)
            # print("red=", ang)
            # print("ang.shape = ", ang.shape) 
    
    
    # Creating contour to track green color 
            
    
    
    contours, hierarchy = cv2.findContours(yellow_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
      
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 300): 
            rect = cv2.minAreaRect(contour) 
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            ang = np.empty((0,2), dtype=int)
            for i in range(4):
                # box[i][0] = int64
                ang = np.append(ang, [[box[i][0], box[i][1]]], axis=0)
                #imageFrame = cv2.circle(imageFrame,(box[i][0], box[i][1]),10,(0,255,255),-1)
            D= ang.mean(axis=0) #원점.
            D = D.ravel()
            D = np.int32(D)

            if(color_count[0]<count):
                turtle_point[0][color_count[0]] = D # 노랑색은 터틀 1번, 해당 터틀봇의 좌표를 turtle_point[0]에 넣음.
                color_count[0] += 1

            if(color_count[0]>=count):
                x_value = np.zeros(color_count[0])
                y_value = np.zeros(color_count[0])

                for i in range(color_count[0]):
                    x_value[i] = turtle_point[0][i][0]
                    mean_x_value = np.mean(x_value)
                    std_x_value = np.std(x_value)
                    threshold = 0.5
                    small_x_value = [value for value in x_value if abs(value - mean_x_value) < threshold * std_x_value]
                    mean_small_x_value = np.mean(small_x_value)
                    
                    y_value[i] = turtle_point[0][i][1]
                    mean_y_value = np.mean(y_value)
                    std_y_value = np.std(y_value)
                    threshold = 0.5
                    small_y_value = [value for value in y_value if abs(value - mean_y_value) < threshold * std_y_value]
                    mean_small_y_value = np.mean(small_y_value) #'numpy.float64'
                if not np.isnan(mean_small_y_value):
                    get_y = int(mean_small_y_value)    
                if not np.isnan(mean_small_x_value):
                    get_x = int(mean_small_x_value)   
                
                
                m_turtle_point[0] = (get_x - zero[0], get_y-zero[1])
                print(m_turtle_point[0])
                imageFrame = cv2.circle(imageFrame,(get_x, get_y),10,(0,255,255),-1)
                color_count[0] = 0
                turtle_point[0] = np.zeros(shape=turtle_point[0].shape)

                

            
            
            
             

  
    # Creating contour to track yellow color 
    contours, hierarchy = cv2.findContours(green_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 300): 
            rect = cv2.minAreaRect(contour) 
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # print(box)
            # print(box.shape)
            # for i in range(4):
            #     imageFrame = cv2.circle(imageFrame,(box[i][0], box[i][1]),10,(0,255,255),-1)

    

    
    for i in range(turtle_num): #real_size에는 [ [ ] (1번은 노랑색 터틀봇), [] (2번은 ...)]
        if(m_turtle_point[i][0] != 0) :
            # m_turtle_point[i][0] -= zero[0]
            # m_turtle_point[i][1] -= zero[1]
            re_turtle_point[i] = np.array([[int(m_turtle_point[i][0])], [int(m_turtle_point[i][1])]] )
            re_turtle_point[i] = np.dot(re_left_side, re_turtle_point[i]) # 원점을 중심, 원점의 x, y 기저 벡터를 통한 새로운 좌표  
            
            real_size[i][0] = real_sqaure_size*re_turtle_point[i][0]/pixel_x_size
            real_size[i][1] = real_sqaure_size*re_turtle_point[i][1]/pixel_y_size  
            
            
            print("노랑색 터틀 봇 좌표 =", real_size[i])
        


    
    #     time.sleep(0.3) 
    
    

    # Program Termination 
    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame) 
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        webcam.release() 
        cv2.destroyAllWindows() 
        break