import numpy as np

blueLower = np.array([120, 0, 0])    # 추출할 색의 하한(BGR)
blueUpper = np.array([250, 140, 10])    # 추출할 색의 상한(BGR)
yellowLower = np.array([0, 190, 230])    # 추출할 색의 하한(BGR)   // 0, 130 ,150
yellowUpper = np.array([240, 250, 250])    # 추출할 색의 상한(BGR)  // 55,250 ,250


class Color_Object:
    def __init__(self,Color):
        self.MyColor = Color

    def get_points(self,points):
        self.points = points
    