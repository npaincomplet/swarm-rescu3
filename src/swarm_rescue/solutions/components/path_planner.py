import numpy as np
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

        dist_transform = cv2.distanceTransform(~obstacle_mask.astype(np.uint8), cv2.DIST_L2, 3)

        # Decreasing penalty within cell_radius, zero beyond
        penalty = np.where(dist_transform <= cell_radius, base_penalty / (dist_transform + 1), 0)
        
        self.costmap[obstacle_mask] = np.inf
        
        self.costmap += penalty
    
    def update(self):
        self.costmap = np.zeros_like(self.occupancy_grid.grid, dtype=float)

        undiscovered_mask = self.occupancy_grid.undiscovered_mask()
        self.costmap[undiscovered_mask] = np.inf

        self.inflate_obstacles()

class PathPlanner:
    def __init__(self, occupancy_grid: OccupancyGrid):
        self.occupancy_grid = occupancy_grid
        self.costmap_grid = CostmapGrid(occupancy_grid)

    def plan_path_to_target(self, start_pos: np.ndarray, target_pos: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Computes the safest path from start_pos to target_pos.
        Returns the path as a list of positions, or None if unreachable.
        """
        start_cell = self.occupancy_grid._conv_world_to_grid(start_pos)
        target_cell = self.occupancy_grid._conv_world_to_grid(target_pos)

        self.costmap_grid.update()

        path = []

        return path