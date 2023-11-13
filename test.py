import cv2

# 색상 범위 설정
lower_orange = (10, 100, 100)
upper_orange = (20, 255, 255)

lower_green = (40, 100, 100)
upper_green = (80, 255, 255)

lower_blue = (90, 50, 50)
upper_blue = (130, 255, 255)

# 이미지 파일을 읽어온다
img = cv2.imread("check_color.jpg", cv2.IMREAD_COLOR)

# BGR to HSV 변환
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 색상 범위를 제한하여 mask 생성
mask_orange = cv2.inRange(img_hsv, lower_orange, upper_orange)
mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

# 원본 이미지를 가지고 Object 추출 이미지로 생성
result_orange = cv2.bitwise_and(img, img, mask=mask_orange)
result_green = cv2.bitwise_and(img, img, mask=mask_green)
result_blue = cv2.bitwise_and(img, img, mask=mask_blue)

# 결과 이미지를 표시
cv2.imshow("Orange", result_blue)

cv2.waitKey(0)
cv2.destroyAllWindows()