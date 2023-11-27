from heapq import heappush as Push
from heapq import heappop as Pop
import robot_class


class TimeAstar:
    def __init__(self,size,robots,MAP) -> None:
        self.size = size
        self.robots = robots.copy()
        self.MAP = MAP.copy()
        self.TimeTable = [[[0,0]*len(robots) for _ in range(size)] for _ in range(size)]
        self.RelativeMap = [[0 for _ in range(size)] for _ in range(size)]

    def distacne(A,B) -> int:
        return abs(A[0]-B[0]) + abs(A[1]-B[1])
    
    def get_robots(self,robots):
        self.robots = robots.copy()

    def get_MAP(self,MAP):
        self.MAP = MAP.copy()

    def Set_obstacle(obstacles) -> None:
        for obstacle in obstacles:
            pass
    def path_tracking(self,coordination) -> list:
        List = []
        Table = self.TimeTable
        tmp = coordination.copy()
        while(self.RelativeMap[tmp[0]][tmp[1]] != tmp):
            
            tmp = self.RelativeMap[tmp[0]][tmp[1]]
        return List
    def Search(self,idx) -> None: # Robot Path finding 
        # Heuristic = Distance  // F = G(현재까지 온 거리) + H(맨하튼 거리)
        Q = []     # [F,coordinate,]
        goal = self.robots[idx].goal
        RelativeMap = self.RelativeMap
        sy,sx = self.robots[idx].coordinate
        RelativeMap[sy][sx] = [sy,sx]
        Push(Q,[self.distance([sy,sx] , goal),[sy,sx],self.robots[idx].direction])
        speed= self.robots[idx].straight
        rotate= self.robots[idx].rotate
        while Q:
            t,cooper,dir = Pop(Q)
            if(cooper == goal):
                break
            for ddir,ku in enumerate([[0,1],[0,-1],[1,0],[-1,0],[0,0]]):
                y = ku[0] + cooper[0]
                x = ku[1] + cooper[1]
                if x < 0 or y < 0 or x > n-1 or y > n-1 or MAP[y][x]== -1: continue
                if ddir == 4:
                    Push(Q,[self.distacne([y,x],goal)+t+1, [y,x], dir])
                else:
                    if ddir ^ dir == 3:
                        st = self.distacne([y,x],goal)+t+speed
                        if all(map(lambda r: st<r[0] or st>r[1], self.TimeTable[y][x])):
                            Push(Q,[st, [y,x], ddir])
                            RelativeMap[y][x] = cooper
                    else:
                        st = self.distacne([y,x],goal)+t+speed+rotate
                        if all(map(lambda r: st<r[0] or st>r[1], self.TimeTable[y][x])):
                            Push(Q,[st, [y,x] , ddir])
                            RelativeMap[y][x] = cooper
        self.robots[idx].get_path(self.path_tracking(goal))
    

n = int(input())
MAP = [[*map(int,input().split())] for _ in range(n)]
robots = [robot_class([0,0],0,1,1),robot_class([0,1],0,1,1)]
astar = TimeAstar(size=n,robots=robots,MAP = MAP)
for i in range(len(robots)):
    astar.Search(i)
    astar.post_turtlebot()

'''
5
0 0 0 0 0    
0 0 0 1 0
0 1 1 1 0
0 0 0 0 0
0 0 0 0 0

'''