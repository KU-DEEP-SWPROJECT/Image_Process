import cv2
import numpy as np
import pyrealsense2
from realsense_depth import *

dc = DepthCamera()

while cv2.waitKey(33)!=ord('q'):
    ret,_,frame = dc.get_frame()
    cv2.imshow("Frame",frame)
cv2.destroyAllWindows()