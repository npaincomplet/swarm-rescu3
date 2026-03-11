import numpy as np
import cv2
from swarm_rescue.simulation.utils.constants import MAX_RANGE_LIDAR_SENSOR
from swarm_rescue.solutions.components.pose import PoseEstimator
from swarm_rescue.solutions.components.astar import *
from swarm_rescue.solutions.utils.dataclasses_config import *
from swarm_rescue.solutions.utils.utils import bresenham

from sklearn.cluster import DBSCAN, KMeans

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
    
    def __str__(self):
        return f"Frontier with {self.size} cells, centroid at pos {self.compute_centroid_pos()}"

    @property
    def size(self):
        return len(self.cells)

    @property
    def positions(self):
        """World positions of the frontier cells."""
        return self._conv_grid_to_world(self.cells)

    def compute_centroid_cell(self):
        if self.size == 0:
            return None
        
        return np.mean(self.cells, axis=0)
    
    def compute_centroid_pos(self):
        centroid = self.compute_centroid_cell()
        if centroid is None:
            return None
        
        return self._conv_grid_to_world(centroid)
    
    def is_large_arc(self) -> bool:
        """
        Return True iff the frontier forms an arc > 180 degrees around its centroid.
        """
        if self.size < 2 * self.MIN_FRONTIER_SIZE:
            return False
        
        centroid = self.compute_centroid_cell()

        dy = self.cells[:, 1] - centroid[1]
        dx = self.cells[:, 0] - centroid[0]
        angles = np.arctan2(dy, dx)
        
        sorted_indices = np.argsort(angles)
        sorted_angles = angles[sorted_indices]

        # Calculate difference between adjacent angles (including the wrap-around)
        diffs = np.diff(sorted_angles)
        wrap_diff = (sorted_angles[0] + 2 * np.pi) - sorted_angles[-1]
        all_diffs = np.append(diffs, wrap_diff)
        max_gap = np.max(all_diffs)

        # A small angular max_gap hints at a circular frontier
        return max_gap < GridParams.FRONTIER_SPLIT_THRESHOLD
    
    def split(self) -> list["Frontier"]:
        """
        Force split the frontier into two smaller frontiers.
        Uses K-Means clustering to ensure spatial continuity.
        """
        if self.size < 2:
            return [self]

        kmeans = KMeans(n_clusters=2, n_init=10).fit(self.cells)
        labels = kmeans.labels_

        f1_cells = self.cells[labels == 0]
        f2_cells = self.cells[labels == 1]

        if len(f1_cells) < self.MIN_FRONTIER_SIZE or len(f2_cells) < self.MIN_FRONTIER_SIZE:
            return [self]

        f1 = Frontier(f1_cells, self._conv_grid_to_world)
        f2 = Frontier(f2_cells, self._conv_grid_to_world)
        return [f1, f2]
    
    def split_if_large_arc(self) -> list["Frontier"]:
        if self.is_large_arc():
            return self.split()
        
        return [self]

class Grid:
    """
    Complex-valued 2D grid
    Real part and imaginary part of every cell are bounded
    """

    def __init__(self,
                 size_area_world,
                 resolution: float):
        self.size_area_world = size_area_world
        self.resolution = resolution

        self.x_max_grid: int = (
            int(self.size_area_world[0] / self.resolution + 0.5))
        self.y_max_grid: int = (
            int(self.size_area_world[1] / self.resolution + 0.5))
        
        # Bounds
        self.clip_min = GridParams.CLIP_MIN
        self.clip_max = GridParams.CLIP_MAX

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid), dtype=complex)

    @property
    def total_cells(self):
        return self.x_max_grid * self.y_max_grid

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
    
    def _clip_and_update_values(self, xs, ys, val):
        """
        All non-constant updates on the grid must use this function to ensure values are bounded
        """
        self.grid[xs, ys] += val
        
        current_values = self.grid[xs, ys]
        clipped_real = np.clip(current_values.real, self.clip_min, self.clip_max)
        clipped_imag = np.clip(current_values.imag, self.clip_min, self.clip_max)
        
        self.grid[xs, ys] = clipped_real + 1j * clipped_imag
        
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

        self._clip_and_update_values(points[0], points[1], val)

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
            if self.cell_in_bounds(grid_coords):
                self._clip_and_update_values(int(x_px), int(y_px), val)
        
        # Multiple points case: shape (n, 2)
        elif grid_coords.ndim == 2:
            x_coords = grid_coords[:, 0]
            y_coords = grid_coords[:, 1]
            
            # Select only points within grid bounds
            mask = (x_coords >= 0) & (x_coords < self.x_max_grid) & \
                   (y_coords >= 0) & (y_coords < self.y_max_grid)
            
            valid_x = x_coords[mask].astype(int)
            valid_y = y_coords[mask].astype(int)
            
            if len(valid_x) > 0:
                self._clip_and_update_values(valid_x, valid_y, val)

    def display(self, title="grid"):
        """
        Screen display of grid and robot pose,
        using opencv (faster than the matplotlib version)
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """
        img = self.grid.T
        if np.iscomplexobj(img):
            img = img.real + (img.real==0) * img.imag
        img = img - img.min()
        img = img / img.max() * 255
        img = np.uint8(img)

        img_color = cv2.applyColorMap(src=img, colormap=cv2.COLORMAP_JET)

        display_size = tuple(int(dim * GridParams.GRID_DISPLAY_RATIO) for dim in self.size_area_world)
        img_color_zoomed = cv2.resize(img_color, display_size, interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow(title, img_color_zoomed)
        cv2.waitKey(1)


class OccupancyGrid(Grid):
    """
    Self updating occupancy grid. Grids are complex-valued.
    For each cell of the grid:
        Real part relates to observations when the drone has access to the gps (quality observation)
        Imaginary part relates to other observations (noisy observation)
    """

    OBSTACLE = GridParams.OBSTACLE
    FREE = GridParams.FREE
    UNDISCOVERED = GridParams.UNDISCOVERED
    MIN_FRONTIER_SIZE = GridParams.MIN_FRONTIER_SIZE
    CLUSTERING_EPSILON = GridParams.CLUSTERING_EPSILON

    def __init__(self,
                 size_area_world,
                 resolution: float,
                 lidar,semantic):
        super().__init__(size_area_world=size_area_world,
                         resolution=resolution)
        self.grid_params = GridParams()

        self.lidar = lidar
        self.semantic = semantic

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
        """
        For each cell:
            If real part is non zero, use it (quality observations)
            Else use imaginary part (noisy observations)
        """
        OBSTACLE_THRESHOLD = GridParams.OBSTACLE_THRESHOLD
        FREE_THRESHOLD = GridParams.FREE_THRESHOLD

        real_part = self.grid.real
        imag_part = self.grid.imag

        use_imag_mask = (real_part == 0)
        
        effective_grid = np.where(use_imag_mask, imag_part, real_part)

        ternary_map = np.zeros_like(real_part, dtype=int)
        ternary_map[effective_grid > OBSTACLE_THRESHOLD] = self.OBSTACLE
        ternary_map[effective_grid < FREE_THRESHOLD] = self.FREE
        ternary_map[effective_grid == 0] = self.UNDISCOVERED
        return ternary_map
    
    def update(self, pose: PoseEstimator, gps_enabled: bool = True):
        """
        Updates the occupancy grid using ray casting algorithm with lidar data.
        Sensor noise : Gaussian(m=0, s=2.5)
        
        Args:
            pose: The current pose of the drone
            gps_enabled: Whether the update is performed with GPS data
        """
        self._update_free_space(pose, gps_enabled)
        
        self._update_obstacles(pose, gps_enabled)

    def _update_free_space(self, pose: PoseEstimator, gps_enabled: bool):
        # Sample lidar data at regular intervals
        lidar_dist = self.lidar.get_sensor_values()[::self.grid_params.EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::self.grid_params.EVERY_N].copy()

        # Ray direction vectors
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        # Clip lidar distances to ensure (probabilistically) true noiseless lidar distance (plus position estimator error) is not exceeded
        clip_dist = self.grid_params.MAX_LIDAR_DIST_CLIP * (lidar_dist / MAX_RANGE_LIDAR_SENSOR)
        confidence_dist = lidar_dist - clip_dist
        np.clip(confidence_dist, 0, MAX_RANGE_LIDAR_SENSOR, out=confidence_dist)

        # Calculate points we are confident form a free line starting from the drone position
        ray_confidence_endpoints = np.column_stack((
            pose.position[0] + np.multiply(confidence_dist, cos_rays),
            pose.position[1] + np.multiply(confidence_dist, sin_rays)
        ))

        update_value = self.grid_params.EMPTY_ZONE_VALUE
        if not gps_enabled:
            update_value = update_value * 1j

        for ray_endpoint in ray_confidence_endpoints:
            cell_endpoint = self._conv_world_to_grid(ray_endpoint)
            x,y = cell_endpoint
            self.add_value_along_line(pose.position, ray_endpoint, update_value)

    def _update_obstacles(self, pose: PoseEstimator, gps_enabled: bool):
        # Sample lidar data at regular intervals
        lidar_dist = self.lidar.get_sensor_values()[::self.grid_params.EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::self.grid_params.EVERY_N].copy()

        # Ray direction vectors
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        # Ray hits an obstacle iff true noiseless lidar distance is less than MAX_RANGE
        # To tackle the added noise we use a safety threshold
        no_obstacle_threshold = MAX_RANGE_LIDAR_SENSOR - self.grid_params.LIDAR_OBSTACLE_MARGIN
        ray_hit_obstacle = lidar_dist < no_obstacle_threshold

        update_value = self.grid_params.OBSTACLE_ZONE_VALUE
        if not gps_enabled:
            update_value = update_value * 1j

        ray_endpoints = np.column_stack((
            pose.position[0] + np.multiply(lidar_dist, cos_rays),
            pose.position[1] + np.multiply(lidar_dist, sin_rays)
        ))

        obstacle_points = ray_endpoints[ray_hit_obstacle]
        self.add_value_to_points(obstacle_points, update_value)
        
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
    
    def _cluster_frontier_cells(self, frontier_cells) -> list[Frontier]:
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
            frontiers.append(Frontier(cluster_points, self._conv_grid_to_world))
            
        return frontiers
    
    def update_frontiers(self):
        # Convert grid to ternary representation (OBSTACLE, FREE, UNDISCOVERED)
        ternary_map = self.to_ternary_map()
        
        # Find frontier cells (boundaries between FREE and UNDISCOVERED)
        frontier_cells = self._identify_frontier_cells(ternary_map)
        
        # No frontiers cells found therefore no frontiers
        if len(frontier_cells) == 0:
            self.frontiers = []
            return []
            
        # Cluster frontier cells using DBSCAN
        raw_frontiers = self._cluster_frontier_cells(frontier_cells)

        refined_frontiers = []
        for frontier in raw_frontiers:
            refined_frontiers.extend(frontier.split_if_large_arc())
        
        self.frontiers = refined_frontiers
        return refined_frontiers

    def delete_frontier_artifacts(self, frontier):
        """
        Set to FRONTIER_ARTIFACT_VALUE (which relates to OBSTACLE) in the grid all cells of frontier
        """
        reset_val = GridParams.FRONTIER_ARTIFACT_RESET_VALUE
        reset_complex = reset_val + reset_val * 1j

        if frontier is not None:
            for cell in frontier.cells:
                self.grid[tuple(cell)] = reset_complex
    
    def _perimeter_cells(self, center, max_radius):
        for r in range(max_radius + 1):
            for dx in range(-r, r + 1):
                yield center + np.array([dx, -r])
                yield center + np.array([dx,  r])
            for dy in range(-r + 1, r):
                yield center + np.array([-r, dy])
                yield center + np.array([ r, dy])

    def find_nearest_free_cell(self, target_cell, max_radius=20):
        # _perimeter_cells is a generator function (yield)
        for cell in self._perimeter_cells(target_cell, max_radius):
            if self.cell_in_bounds(cell) and self.is_free(self._get_effective_value(self.grid[tuple(cell)])):
                return cell
            
        return None
    
    def check_line_of_sight(self, start_coords, end_coords):
        """
        Check if there is a direct line of sight between start and end coordinates.
        Returns True if the line does not pass through obstacles, False otherwise.
        Input in world coordinates.
        """
        start_cell = self._conv_world_to_grid(start_coords)
        end_cell = self._conv_world_to_grid(end_coords)

        if not self.cell_in_bounds(start_cell) or not self.cell_in_bounds(end_cell):
            return False
        
        for cell in bresenham(start_cell, end_cell):
            if not self.cell_in_bounds(cell):
                return False
            if self.is_obstacle(self._get_effective_value(self.grid[tuple(cell)])):
                return False
            
        return True
    
    def mark_kill_zone_as_obstacle(self, kill_position):
        """Mark a kill zone location as an inflated impassable obstacle"""
        center_cell = self._conv_world_to_grid(kill_position)
        x_center, y_center = center_cell

        mark_cell_radius = self.grid_params.KILL_ZONE_CELL_RADIUS
        mark_cell_value = self.grid_params.KILL_ZONE_VALUE
        cell = np.empty(2, dtype=int)
        
        for dx in range(-mark_cell_radius, mark_cell_radius + 1):
            for dy in range(-mark_cell_radius, mark_cell_radius + 1):
                cell[0] = x_center + dx
                cell[1] = y_center + dy
                if self.cell_in_bounds(cell):
                    self._clip_and_update_values(cell[0], cell[1], mark_cell_value)

    def mark_unreachable_undiscovered_as_obstacles(self):
        undiscovered_mask = self.undiscovered_mask().astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(undiscovered_mask, connectivity=4)
        
        free_mask_uint8 = self.free_mask().astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        
        # Skip label 0 (background)
        for label in range(1, num_labels):
            component_mask = (labels == label).astype(np.uint8)
            dilated_component = cv2.dilate(component_mask, kernel, iterations=1)
            intersection = dilated_component & free_mask_uint8
            is_reachable = np.any(intersection)
            region_size = stats[label, cv2.CC_STAT_AREA]
            
            if not is_reachable and region_size > self.grid_params.MIN_UNREACHABLE_REGION_SIZE:
                component_coords = np.argwhere(component_mask)
                obs_val = self.grid_params.UNREACHABLE_REGION_VALUE
                self.grid[component_coords[:, 0], component_coords[:, 1]] = obs_val + obs_val * 1j
    
    def compute_exploration_score(self):
        """Between 0 and 1."""
        self.mark_unreachable_undiscovered_as_obstacles()
        # Cells with values 0 (initialization) are assumed not to have been visited
        # Check if both real AND imag are 0
        explored_cells = np.sum(self.grid != 0)
        return explored_cells / self.total_cells

    def merge_grids(self, other_grid):
        self.grid = (self.grid + other_grid)/2
    
    def _get_effective_value(self, cell_value):
        if np.iscomplexobj(cell_value):
            if cell_value.real != 0:
                return cell_value.real
            return cell_value.imag
        return cell_value

    def _get_effective_grid(self):
        real_part = self.grid.real
        imag_part = self.grid.imag
        use_imag_mask = (real_part == 0)
        return np.where(use_imag_mask, imag_part, real_part)

    def is_free(self, cell_value):
        val = self._get_effective_value(cell_value)
        return val < GridParams.FREE_THRESHOLD
    
    def is_obstacle(self, cell_value):
        val = self._get_effective_value(cell_value)
        return val > GridParams.OBSTACLE_THRESHOLD

    def is_undiscovered(self, cell_value):
        val = self._get_effective_value(cell_value)
        return GridParams.FREE_THRESHOLD <= val <= GridParams.OBSTACLE_THRESHOLD

    def pos_radius_has_obstacle(self, pos, cell_radius):
        center_cell = self._conv_world_to_grid(pos)
        x_center, y_center = center_cell
        
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell = np.array([x_center + dx, y_center + dy])
                if self.cell_in_bounds(cell):
                    cell_value = self.grid[tuple(cell)]
                    if self.is_obstacle(cell_value):
                        return True
        return False

    def free_mask(self):
        eff = self._get_effective_grid()
        return eff < GridParams.FREE_THRESHOLD

    def obstacle_mask(self):
        eff = self._get_effective_grid()
        return eff > GridParams.OBSTACLE_THRESHOLD
    
    def undiscovered_mask(self):
        eff = self._get_effective_grid()
        return np.logical_and(GridParams.FREE_THRESHOLD <= eff, eff <= GridParams.OBSTACLE_THRESHOLD)