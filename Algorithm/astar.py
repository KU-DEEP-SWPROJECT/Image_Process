import heapq

class Node:
    def __init__(self, row, col, cost, heuristic):
        self.row = row
        self.col = col
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    visited = set()
    heap = [Node(start[0], start[1], 0, 0)]
    
    while heap:
        current = heapq.heappop(heap)
        
        if (current.row, current.col) == goal:
            # Reached the goal
            return current.cost
        
        if (current.row, current.col) in visited:
            continue

        visited.add((current.row, current.col))

        # Generate neighbors
        neighbors = [
            (current.row - 1, current.col),
            (current.row + 1, current.col),
            (current.row, current.col - 1),
            (current.row, current.col + 1)
        ]

        for neighbor_row, neighbor_col in neighbors:
            if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols and grid[neighbor_row][neighbor_col] != -1:
                if (neighbor_row, neighbor_col) not in visited:
                    neighbor_cost = current.cost + grid[neighbor_row][neighbor_col]
                    neighbor_heuristic = heuristic(neighbor_row, neighbor_col, goal[0], goal[1])
                    heapq.heappush(heap, Node(neighbor_row, neighbor_col, neighbor_cost, neighbor_heuristic))

    # Goal is not reachable
    return -1

def heuristic(row1, col1, row2, col2):
    # Manhattan distance heuristic
    return abs(row1 - row2) + abs(col1 - col2)

# Example usage
grid = [
    [1, 1, 1, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]

start = (0, 0)
goal = (3, 4)

result = astar(grid, start, goal)

if result != -1:
    print(f"Shortest path cost: {result}")
else:
    print("No path to the goal")
