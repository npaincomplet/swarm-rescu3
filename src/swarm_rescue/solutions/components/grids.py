import numpy as np
import cv2
from swarm_rescue.simulation.utils.constants import MAX_RANGE_LIDAR_SENSOR
from solutions.components.pose import Pose
from solutions.components.astar import *
from solutions.utils.dataclasses_config import *

from sklearn.cluster import DBSCAN

class Grid:
    """Simple grid"""

    def __init__(self,
                 size_area_world,
                 resolution: float):
        self.size_area_world = size_area_world
        self.resolution = resolution

        self.x_max_grid: int = (
            int(self.size_area_world[0] / self.resolution + 0.5))
        self.y_max_grid: int = (
            int(self.size_area_world[1] / self.resolution + 0.5))

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))

    def _conv_world_to_grid(self, world_coords):
        """
        Convert world coordinates to grid cell indices.
        
        Args:
            world_coords: Either:
                - A numpy array of shape (2,) containing [world_x, world_y] coordinates
                - A numpy array of shape (n, 2) containing multiple [world_x, world_y] coordinates
        
        Returns:
            A numpy array with the same shape structure as input, containing integer grid coordinates
        """
        world_coords = world_coords.copy()
        
        # Single point case: shape (2,)
        if world_coords.ndim == 1:
            world_coords[1] = -world_coords[1]  # Invert y-axis for grid coordinates
            grid_cell = (world_coords + np.array(self.size_area_world) / 2) / self.resolution
            return grid_cell.astype(int)
        
        # Multiple points case: shape (n, 2)
        elif world_coords.ndim == 2:
            world_coords[:, 1] = -world_coords[:, 1]  # Invert y-axis for grid coordinates
            grid_cells = (world_coords + np.array(self.size_area_world) / 2) / self.resolution
            return grid_cells.astype(int)
        
        else:
            raise ValueError(f"Invalid shape for world_coords: {world_coords.shape}")

    def _conv_grid_to_world(self, grid_cell):
        """
        Convert grid cell to world coordinates.
        
        Args:
            grid_cell: Either:
                - A numpy array of shape (2,) containing [grid_x, grid_y] cell
                - A numpy array of shape (n, 2) containing multiple [grid_x, grid_y] cells
        
        Returns:
            A numpy array with the same shape structure as input, containing float world coordinates
        """
        grid_cell = grid_cell.copy()
        
        # Single point case: shape (2,)
        if grid_cell.ndim == 1:
            world_coords = -np.array(self.size_area_world, dtype=float) / 2 + grid_cell * self.resolution
            world_coords[1] = -world_coords[1]  # Invert y-axis for world coordinates
            return world_coords
        
        # Multiple points case: shape (n, 2)
        elif grid_cell.ndim == 2:
            world_coords = -np.array(self.size_area_world, dtype=float) / 2 + grid_cell * self.resolution
            world_coords[:, 1] = -world_coords[:, 1]  # Invert y-axis for world coordinates
            return world_coords
        
        else:
            raise ValueError(f"Invalid shape for grid_cell: {grid_cell.shape}")
        
    def cell_in_bounds(self, cell_coords):
        """
        Check if the given cell coordinates are within the grid bounds.
        
        Args:
            cell_coords: A numpy array of shape (2,) containing [grid_x, grid_y] cell coordinates
        
        Returns:
            bool: True if the cell is within bounds, False otherwise
        """
        x, y = cell_coords
        return 0 <= x < self.x_max_grid and 0 <= y < self.y_max_grid

    def add_value_along_line(self, start_coords, end_coords, val):
        """
        Add a value to a line of points using Bresenham algorithm. 
        Input in world coordinates.
        """
        start_cell = self._conv_world_to_grid(start_coords)
        x_start, y_start = start_cell
        
        end_cell = self._conv_world_to_grid(end_coords)
        x_end, y_end = end_cell

        if not self.cell_in_bounds(start_cell):
            return

        if not self.cell_in_bounds(end_cell):
            return

        # Bresenham line drawing
        dx = x_end - x_start
        dy = y_end - y_start
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x_start, y_start = y_start, x_start
            x_end, y_end = y_end, x_end
        # swap start and end points if necessary and store swap state
        if x_start > x_end:
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start
        dx = x_end - x_start  # recalculate differentials
        dy = y_end - y_start  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y_start < y_end else -1
        # iterate over bounding box generating points between start and end
        y = y_start
        points = []
        for x in range(x_start, x_end + 1):
            coord = [y, x] if is_steep else [x, y]
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        points = np.array(points).T

        # add value to the points
        self.grid[points[0], points[1]] += val

    def add_value_to_points(self, points_coords, val):
        """
        Add a value to an array of points in the grid.
        
        Args:
            points_coords: Either:
                - A numpy array of shape (2,) containing [world_x, world_y] coordinates
                - A numpy array of shape (n, 2) containing multiple [world_x, world_y] coordinates
            val: value to add to the cells of the points
        """
        # Convert world coordinates to grid coordinates
        grid_coords = self._conv_world_to_grid(points_coords)
        
        # Single point case: shape (2,)
        if grid_coords.ndim == 1:
            x_px, y_px = grid_coords
            if 0 <= x_px < self.x_max_grid and 0 <= y_px < self.y_max_grid:
                self.grid[int(x_px), int(y_px)] += val
        
        # Multiple points case: shape (n, 2)
        elif grid_coords.ndim == 2:
            # Select only points within grid bounds
            select = np.logical_and(
                np.logical_and(grid_coords[:, 0] >= 0, grid_coords[:, 0] < self.x_max_grid),
                np.logical_and(grid_coords[:, 1] >= 0, grid_coords[:, 1] < self.y_max_grid)
            )
            
            valid_coords = grid_coords[select]
            if len(valid_coords) > 0:
                # Convert to integer indices and add values
                x_indices = valid_coords[:, 0].astype(int)
                y_indices = valid_coords[:, 1].astype(int)
                self.grid[x_indices, y_indices] += val

    def display(self, grid_to_display: np.ndarray,
                robot_pose: Pose, title="grid"):
        """
        Screen display of grid and robot pose,
        using opencv (faster than the matplotlib version)
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """
        img = grid_to_display.T
        img = img - img.min()
        img = img / img.max() * 255
        img = np.uint8(img)
        img_color = cv2.applyColorMap(src=img, colormap=cv2.COLORMAP_JET)
        
        cv2.imshow(title, img_color)
        cv2.waitKey(1)


class OccupancyGrid(Grid):
    """Self updating occupancy grid"""

    OBSTACLE = GridParams.OBSTACLE
    FREE = GridParams.FREE
    UNDISCOVERED = GridParams.UNDISCOVERED
    MIN_FRONTIER_SIZE = GridParams.MIN_FRONTIER_SIZE
    CLUSTERING_EPSILON = GridParams.CLUSTERING_EPSILON

    class Frontier:
        """
        Represents a frontier - a collection of adjacent cells that form a boundary 
        between explored (FREE) and unexplored (UNDISCOVERED) areas of the grid.
        All cell coordinates are in grid coordinate system.
        """
        MIN_FRONTIER_SIZE = GridParams.MIN_FRONTIER_SIZE

        def __init__(self, cells, _conv_grid_to_world=None):
            """
            :param cells: List of arrays [array([x, y]), ...] representing the grid coordinates of the frontier cells.
            """
            self.cells = cells
            self._conv_grid_to_world = _conv_grid_to_world

        def compute_centroid_cell(self):
            if self.cells.size == 0:
                return None
            
            return np.mean(self.cells, axis=0)
        
        def compute_centroid_pos(self):
            centroid = self.compute_centroid_cell()
            if centroid is None:
                return None
            
            return self._conv_grid_to_world(centroid)

        def size(self):
            return len(self.cells)

    def __init__(self,
                 size_area_world,
                 resolution: float,
                 lidar,semantic):
        super().__init__(size_area_world=size_area_world,
                         resolution=resolution)
        self.grid_params = GridParams()

        self.lidar = lidar
        self.semantic = semantic

        self.zoomed_grid = self.grid.copy()

        self._init_world_borders()

        self.frontiers = []

    def _init_world_borders(self):
        """
        Set the value of all border cells to WORLD_BORDERS_VALUE so they are considered as obstacles.
        """
        WORLD_BORDERS_VALUE = GridParams.WORLD_BORDERS_VALUE
        self.grid[[0, -1], :] = WORLD_BORDERS_VALUE
        self.grid[:, [0, -1]] = WORLD_BORDERS_VALUE
    
    def to_ternary_map(self):
        OBSTACLE_THRESHOLD = GridParams.OBSTACLE_THRESHOLD
        FREE_THRESHOLD = GridParams.FREE_THRESHOLD

        ternary_map = np.zeros_like(self.grid, dtype=int)
        ternary_map[self.grid > OBSTACLE_THRESHOLD] = self.OBSTACLE
        ternary_map[self.grid < FREE_THRESHOLD] = self.FREE
        ternary_map[self.grid == 0] = self.UNDISCOVERED
        return ternary_map
    
    def update(self, pose: Pose):
        """
        Updates the occupancy grid using ray casting algorithm with lidar data.
        Sensor noise : Gaussian(m=0, s=2.5)
        
        Args:
            pose: The current pose of the drone
        """
        self._update_free_space(pose)
        
        self._update_obstacles(pose)

        self.grid = np.clip(self.grid, self.grid_params.THRESHOLD_MIN, self.grid_params.THRESHOLD_MAX)
        
        self._update_zoomed_grid()

    def _update_free_space(self, pose):
        # Sample lidar data at regular intervals
        lidar_dist = self.lidar.get_sensor_values()[::self.grid_params.EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::self.grid_params.EVERY_N].copy()

        # Ray direction vectors
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        # Clip lidar distances to ensure (probabilistically) true noiseless lidar distance is not exceeded
        confidence_dist = lidar_dist - self.grid_params.LIDAR_DIST_CLIP
        np.clip(confidence_dist, 0, MAX_RANGE_LIDAR_SENSOR)

        # Calculate points we are confident form a free line starting from the drone position
        ray_confidence_endpoints = np.column_stack((
            pose.position[0] + np.multiply(confidence_dist, cos_rays),
            pose.position[1] + np.multiply(confidence_dist, sin_rays)
        ))

        for ray_endpoint in ray_confidence_endpoints:
            cell_endpoint = self._conv_world_to_grid(ray_endpoint)
            x,y = cell_endpoint
            if 210<=x and x<=260 and 10<=y and y<=50:
                print(self._conv_world_to_grid(pose.position))
                print(lidar_dist[np.where((ray_confidence_endpoints == ray_endpoint).all(axis=1))[0][0]])
            self.add_value_along_line(pose.position, ray_endpoint, self.grid_params.EMPTY_ZONE_VALUE)

    def _update_obstacles(self, pose):
        # Sample lidar data at regular intervals
        lidar_dist = self.lidar.get_sensor_values()[::self.grid_params.EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::self.grid_params.EVERY_N].copy()

        # Ray direction vectors
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        # Ray hits an obstacle iff true noiseless lidar distance is less than MAX_RANGE
        # To tackle the added noise we use a safety threshold
        no_obstacle_threshold = MAX_RANGE_LIDAR_SENSOR - self.grid_params.LIDAR_DIST_CLIP
        hit_obstacle = lidar_dist < no_obstacle_threshold

        ray_endpoints = np.column_stack((
            pose.position[0] + np.multiply(lidar_dist, cos_rays),
            pose.position[1] + np.multiply(lidar_dist, sin_rays)
        ))

        obstacle_points = ray_endpoints[hit_obstacle]
        self.add_value_to_points(obstacle_points, self.grid_params.OBSTACLE_ZONE_VALUE)

    def _update_zoomed_grid(self):
        zoomed_grid = self.grid.copy()
        
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)
        
    def _identify_frontier_cells(self, ternary_map):
        """
        Identifies cells that form the boundary between explored (FREE) and 
        unexplored (UNDISCOVERED) areas.

        Returns:
            numpy.ndarray: Array of [x, y] coordinates representing frontier cells
        """
        # Calculate differences along X and Y axes to find boundaries
        diff_x = np.diff(ternary_map, axis=1)
        diff_y = np.diff(ternary_map, axis=0)

        # Detect boundaries between FREE and UNDISCOVERED cells (absolute difference of 2)
        boundaries_x = np.abs(diff_x) == 2
        boundaries_y = np.abs(diff_y) == 2

        # Combine horizontal and vertical boundaries with padding to maintain original shape
        boundaries_map = np.pad(boundaries_x, ((0, 0), (0, 1))) | np.pad(boundaries_y, ((0, 1), (0, 0)))
        
        # Get grid coordinates of all frontier cells
        return np.argwhere(boundaries_map)
    
    def _cluster_frontier_cells(self, frontier_cells):
        # Apply DBSCAN clustering
        db = DBSCAN(
            eps=self.CLUSTERING_EPSILON, 
            min_samples=self.MIN_FRONTIER_SIZE
        ).fit(frontier_cells)
        
        labels = db.labels_
        frontiers = []
        
        # Process each cluster
        for label_val in set(labels):
            if label_val == -1:  # Skip noise points
                continue
                
            # Get points belonging to this cluster
            cluster_points = frontier_cells[labels == label_val]
            
            # Create frontier object
            frontiers.append(self.Frontier(cluster_points, self._conv_grid_to_world))
            
        return frontiers
    
    def update_frontiers(self):
        # Convert grid to ternary representation (OBSTACLE, FREE, UNDISCOVERED)
        ternary_map = self.to_ternary_map()
        
        # Find frontier cells (boundaries between FREE and UNDISCOVERED)
        frontier_cells = self._identify_frontier_cells(ternary_map)
        
        # No frontiers cells found therefore no frontiers
        if len(frontier_cells) == 0:
            return []
            
        # Cluster frontier cells using DBSCAN
        frontiers = self._cluster_frontier_cells(frontier_cells)
        
        self.frontiers = frontiers
        return frontiers

    def delete_frontier_artifacts(self, frontier):
        """
        Set to THRESHOLD_MAX (which relates to OBSTACLE) in the grid all cells of frontier
        """
        print("Deleting frontier artifacts")
        if frontier is not None:
            for cell in frontier.cells:
                self.grid[cell] = GridParams.FRONTIER_ARTIFACT_RESET_VALUE

    def compute_safest_path(self, start_pos, target_pos):
        """
        Returns the path, if it exists, that joins drone's position to target_cell
        If the target cell is not free, finds the nearest free cell and routes to that instead
        start_cell, target_cell : GRID COORDINATES
        """
        MAP = self.to_ternary_map()

        start_cell = self._conv_world_to_grid(start_pos)
        target_cell = self._conv_world_to_grid(target_pos)
        target_cell = self._find_nearest_free_cell(MAP, target_cell)

        start_x, start_y = start_cell
        end_x, end_y = target_cell
        path = a_star_search(MAP, (start_x, start_y), (end_x, end_y))

        if path:
            path_simplified = self.simplify_path(path, MAP) or [start_cell]
            return [np.array(self._conv_grid_to_world(np.array([x,y]))) for x, y in path_simplified]
        
        return None
    
    def _find_nearest_free_cell(self, map_array, target_cell, max_radius=20):
        """
        Args:
            map_array: Ternary map representation
            target_cell: Target cell coordinates [x, y]
            max_radius: Maximum search radius
            
        Returns:
            numpy.ndarray: Array of [x, y] coordinates representing nearest free cell, or None if not found
        """
        # Expand radius until a free cell is found or max_radius is reached
        for r in range(0, max_radius + 1):  # r=0 is the target cell itself
            for dx in range(-r, r + 1):
                for dy in (-r,r):
                    possible_cell = target_cell + np.array([dx, dy])
                    if self.cell_in_bounds(possible_cell):
                        if map_array[tuple(possible_cell)] == self.FREE:
                            return possible_cell
                        
            for dy in range(-r, r + 1):
                for dx in (-r,r):
                    possible_cell = target_cell + np.array([dx, dy])
                    if self.cell_in_bounds(possible_cell):
                        if map_array[tuple(possible_cell)] == self.FREE:
                            return possible_cell
        
        return None

    def simplify_path(self, path, MAP):
        path_simplified = simplify_collinear_points(path)
        path_line_of_sight = simplify_by_line_of_sight(path_simplified, MAP)
        return ramer_douglas_peucker(path_line_of_sight, 0.5)

    def merge_grids(self, other_grid):
        self.grid = (self.grid + other_grid)/2

    def free_mask(self):
        return self.grid < GridParams.FREE_THRESHOLD

    def obstacle_mask(self):
        return self.grid > GridParams.OBSTACLE_THRESHOLD
    
    def undiscovered_mask(self):
        return GridParams.FREE_THRESHOLD <= self.grid <= GridParams.OBSTACLE_THRESHOLD