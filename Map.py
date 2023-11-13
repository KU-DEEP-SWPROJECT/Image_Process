import numpy as np

MAP = [[0]*480 for _ in range(640)] # 640 * 480 MAP
robots = [[-1,-1]]*4                # robots
def init():
    global MAP,robots
    MAP = [[0]*480 for _ in range(640)]
    robots = [[-1,-1]]*4

def input_robots(num, xy):

    if all(robots[num-1]) > -1: # 원래 값이 있었다면 그 자리 비우기
        y = robots[num-1][0]
        x = robots[num-1][1]
        MAP[y][x] = 0

    robots[num-1] = xy
    print(xy)
    y = xy[0]; x=xy[1]            # 로봇 자리 갱신 해줌.
    print(y,x)
    MAP[y][x] = num       # MAP 에 넣어줌

def input_obstacle(obstacles):   # 장애물 MAP에 넣기
    for obstacle in obstacles: 
        for y,x in obstacle:
            MAP[y][x] = -1


def print_map():
    for y in range(640):
        for x in range(480):
            print(MAP[y][x],end=' ')
        print()
