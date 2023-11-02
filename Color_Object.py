import numpy as np

blueLower = np.array([120, 0, 0])    # 추출할 색의 하한(BGR)
blueUpper = np.array([250, 100, 10])    # 추출할 색의 상한(BGR)
yellowLower = np.array([0, 150, 190])    # 추출할 색의 하한(BGR)   // 0, 130 ,150
yellowUpper = np.array([100, 220, 250])    # 추출할 색의 상한(BGR)  // 55,250 ,250

class Color_Object:
    def __init__(self,Color):
        self.MyColor = Color

    def get_points(self,points):
        self.points = points
    def get_contours(self,Fined_Contour):
        self.contours = Fined_Contour[0]
        center = np.mean(self.contours,axis=0)
        center = np.int0(center)
        self.center = center