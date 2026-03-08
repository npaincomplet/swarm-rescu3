import numpy as np
import cv2
from swarm_rescue.solutions.components.astar import a_star
from swarm_rescue.solutions.utils.utils import bresenham
from typing import Optional, List
from swarm_rescue.solutions.components.grids import *
from swarm_rescue.solutions.utils.dataclasses_config import PathPlanningParams

class CostmapGrid():
    """
    Grid of cost values for path planning
    Costs are positive values, higher cost = less desirable
    An obstacle has infinite cost
    """
    def __init__(self,
                 occupancy_grid: OccupancyGrid):
        self.occupancy_grid = occupancy_grid
        self.costmap = np.zeros_like(occupancy_grid.grid, dtype=float)

    def inflate_obstacles(self, base_penalty, cell_radius):
        """Set obstacles as np.inf and inflate them with decreasing penalty"""
        obstacle_mask = self.occupancy_grid.obstacle_mask()

        free = (~obstacle_mask).astype(np.uint8)
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 3)


        # Decreasing penalty within cell_radius, zero beyond
        penalty = np.where(dist <= cell_radius, base_penalty / (dist + 0.01) ** 2, 0)
        
        self.costmap[obstacle_mask] = np.inf
        
        self.costmap += penalty
    
    def update(self):
        self.costmap.fill(PathPlanningParams.CONSTANT_COST)

        undiscovered_mask = self.occupancy_grid.undiscovered_mask()
        self.costmap[undiscovered_mask] = np.inf

        self.inflate_obstacles(PathPlanningParams.OBSTACLE_BASE_PENALTY, PathPlanningParams.GRID_INFLATION_RADIUS)

class PathSimplifier:
    @staticmethod
    def simplify(path, costmap, penalty_allowance=0.0):
        if path is None or len(path) <= 2:
            return path

        path = PathSimplifier._simplify_by_direction(path)
        path = PathSimplifier._simplify_by_los(path, costmap, penalty_allowance)

        return path

    @staticmethod
    def _simplify_by_direction(path):
        d = np.diff(path, axis=0)
        d = np.sign(d)

        keep = [0]
        prev = d[0]
        for i in range(1, len(d)):
            if not np.array_equal(d[i], prev):
                keep.append(i)
                prev = d[i]
        keep.append(len(path) - 1)

        return path[np.array(keep, dtype=int)]
    
    @staticmethod
    def _simplify_by_los(path, costmap, penalty_allowance):
        simplified = []
        i = 0
        n = len(path)

        while i < n:
            simplified.append(path[i])
            if i == n - 1:
                break

            j = n - 1
            while j > i + 1:
                if PathSimplifier._has_los(path[i], path[j], costmap, penalty_allowance):
                    break
                j -= 1

            i = j

        return np.asarray(simplified, dtype=path.dtype)

    @staticmethod
    def _has_los(a: np.ndarray, b: np.ndarray, costmap, penalty_allowance):
        for r, c in bresenham(a, b):
            if costmap[r, c] > penalty_allowance:
                return False
            
        return True

class PathPlanner:
    def __init__(self, occupancy_grid: OccupancyGrid):
        self.occupancy_grid = occupancy_grid
        self.costmap_grid = CostmapGrid(occupancy_grid)

    def plan_path_to_target(self, start_pos: np.ndarray, target_pos: np.ndarray, holds_wounded = False) -> Optional[List[np.ndarray]]:
        """
        Computes the safest path from start_pos to target_pos.
        Returns the path as a list of positions, or None if unreachable.
        """
        start_cell = self.occupancy_grid._conv_world_to_grid(start_pos)
        start_cell = self.occupancy_grid.find_nearest_free_cell(start_cell)

        target_cell = self.occupancy_grid._conv_world_to_grid(target_pos)
        target_cell = self.occupancy_grid.find_nearest_free_cell(target_cell)

        self.costmap_grid.update()

        path = a_star(self.costmap_grid.costmap, start_cell, target_cell)

        if holds_wounded:
            path = PathSimplifier.simplify(path, self.costmap_grid.costmap, penalty_allowance=PathPlanningParams.CAUTION_PENALTY_ALLOWANCE)
        else:
            path = PathSimplifier.simplify(path, self.costmap_grid.costmap, penalty_allowance=PathPlanningParams.SHORTCUT_PENALTY_ALLOWANCE)

        world_path = [self.occupancy_grid._conv_grid_to_world(cell) for cell in path] if path.size > 0 else None

        return world_path