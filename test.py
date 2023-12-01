from shapely.geometry import Point, Polygon
import numpy as np

def fill_burden(points):
    
    polygon = Polygon(points)
    min_x, min_y, max_x, max_y = polygon.bounds
    x_size = max_x - min_x+1
    y_size = max_y - min_y+1
    size = x_size * y_size
    square = np.zeros( ( int(size), 2), dtype=int)
    square_cnt = 0

    for x in range(int(min_x), int(max_x) + 1):
        for y in range(int(min_y), int(max_y) + 1):
            point = Point(x, y)
            if polygon.contains(point):
                square[square_cnt][0] =x
                square[square_cnt][1] =y
                square_cnt += 1
            
            if polygon.exterior.contains(point):
                square[square_cnt][0] =x
                square[square_cnt][1] =y
                square_cnt += 1
            
            
    
    real_square = np.zeros( (square_cnt, 2) , dtype=int)
    for i in range(square_cnt):
        real_square[i] = square[i]

    return real_square


get_burden = ########### 리스트 형식으로 된 4개의 값들 넣으면 됨.

rec_pos = fill_burden(get_burden)

map_size = 1000
map = np.zeros( (map_size, map_size) ,dtype=int)


for i in range(len(rec_pos)):
    map[rec_pos[i][0]][rec_pos[i][1]] = -1



