import numpy as np
import heapq
import math

def octile_heuristic(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    d1 = 1.0
    d2 = math.sqrt(2)
    return d1 * (dx + dy) + (d2 - 2 * d1) * min(dx, dy)

def a_star(grid, start_cell, target_cell):
    start = tuple(start_cell)
    goal = tuple(target_cell)
    
    rows, cols = grid.shape
    open_heap = []
    heapq.heappush(open_heap, (0, 0, start))
    
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    # 8-connectivity: (dx, dy, cost_multiplier)
    neighbors = [
        (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
        (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))
    ]
    
    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        
        if current == goal:
            break
            
        for dx, dy, step_cost in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                cell_cost = grid[neighbor[0], neighbor[1]]
                if np.isinf(cell_cost):
                    continue
                
                new_cost = cost_so_far[current] + cell_cost * step_cost
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + octile_heuristic(goal, neighbor)
                    heapq.heappush(open_heap, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
                    
    if goal not in came_from:
        return np.array([])

    # Reconstruct path
    path = []
    current = goal
    while current is not None:
        path.append(np.array(current))
        current = came_from[current]
    return np.array(path[::-1])