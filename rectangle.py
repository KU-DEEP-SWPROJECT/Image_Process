import cv2
import numpy as np
img = cv2.imread("rect2.png")

bgrLower = (0, 0, 0)    # 추출할 색의 하한(BGR)
bgrUpper = (10, 176, 250)    # 추출할 색의 상한(BGR)
img_mask = cv2.inRange(img, bgrLower, bgrUpper) # BGR로 부터 마스크를 작성
res = cv2.bitwise_and(img, img, mask=img_mask)
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
contours, hierachy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
f = open("rect.txt",'w')
data = contours
print(data)

for contour in contours:
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(img, [approx], -1, (0,255,0), 3)
    D= approx.mean(axis=0)
    D = D.ravel()
    D = np.int32(D)
    print(contour, D)
    cv2.circle(img,D,10,(255,0,0),-1)
    
cv2.imshow("Img",img)
cv2.waitKey(0)
f.close()