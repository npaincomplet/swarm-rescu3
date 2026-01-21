import numpy as np
import cv2
from solutions.components.astarv2 import a_star
from typing import Optional, List
from solutions.components.grids import *
from solutions.utils.dataclasses_config import PathPlanningParams

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

class PathPlanner:
    def __init__(self, occupancy_grid: OccupancyGrid):
        self.occupancy_grid = occupancy_grid
        self.costmap_grid = CostmapGrid(occupancy_grid)

    def plan_path_to_target(self, start_pos: np.ndarray, target_pos: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Computes the safest path from start_pos to target_pos.
        Returns the path as a list of positions, or None if unreachable.
        """
        # Needs nearest free cell logic
        start_cell = self.occupancy_grid._conv_world_to_grid(start_pos)
        start_cell = self.occupancy_grid.find_nearest_free_cell(start_cell)

        target_cell = self.occupancy_grid._conv_world_to_grid(target_pos)
        target_cell = self.occupancy_grid.find_nearest_free_cell(target_cell)

        self.costmap_grid.update()

        path = a_star(self.costmap_grid.costmap, start_cell, target_cell)
        world_path = [self.occupancy_grid._conv_grid_to_world(cell) for cell in path] if path.size > 0 else None

        return world_path