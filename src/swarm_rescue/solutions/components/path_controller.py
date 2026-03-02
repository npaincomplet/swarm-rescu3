import numpy as np

class PathController:
    def __init__(self, path_params, rotation_pid, lateral_pid, forward_pid, ray_angles):
        self.path_params = path_params
        self.rotation_pid = rotation_pid
        self.lateral_pid = lateral_pid
        self.forward_pid = forward_pid
        
        self.index_current_waypoint = 0
        self.initial_point_path = np.zeros(2)
        self.finished_path = True
        self.path = []
        self.path_grid = []

        self._init_lateral_offset_mask(ray_angles)
        self._init_longitudinal_offset_mask(ray_angles)
    
    def _init_lateral_offset_mask(self, ray_angles):
        """
        Mask used on lidar_values for obstacle avoidance (lateral).
        """
        lateral_offset_mask = np.zeros_like(ray_angles)
        
        # Set -1 (translate right laterally) for left side rays
        lateral_offset_mask[ray_angles >= 0] = -1
        
        # Set +1 (translate left laterally) for right side rays
        lateral_offset_mask[ray_angles <= 0] = 1

        self.lateral_offset_mask = lateral_offset_mask

    def _init_longitudinal_offset_mask(self, ray_angles):
        """
        Mask used on lidar_values for obstacle avoidance (forward/backward).
        """
        longitudinal_offset_mask = np.zeros_like(ray_angles)
        
        # Front rays defined by abs(angle) < pi/2 -> push backward (-1)
        longitudinal_offset_mask[np.abs(ray_angles) < np.pi/2] = -1
        
        # Back rays defined by abs(angle) >= pi/2 -> push forward (+1)
        longitudinal_offset_mask[np.abs(ray_angles) >= np.pi/2] = 1
        
        self.longitudinal_offset_mask = longitudinal_offset_mask
    
    def reset_path(self):
        self.finished_path = True
        self.index_current_waypoint = 0
        self.path = []
        self.path_grid = []
        
    def set_path(self, path, initial_point=None):
        if path is not None and len(path) > 0:
            self.path = path
            self.index_current_waypoint = 0
            self.finished_path = False
            if initial_point is not None:
                self.initial_point_path = initial_point
    
    def is_near_waypoint(self, waypoint, current_position, speed):
        distance_to_waypoint = np.linalg.norm(waypoint - current_position)
        
        return (distance_to_waypoint < self.path_params.DISTANCE_CLOSE_WAYPOINT and
                speed < self.path_params.SPEED_CLOSE_WAYPOINT)
    
    def follow_path(self, current_position, current_orientation, odometer_values, lidar_values, ray_angles, found_and_near_wounded=False):
        if self.finished_path or not self.path:
            return None
            
        if self.is_near_waypoint(self.path[self.index_current_waypoint], current_position, np.linalg.norm(odometer_values)):
            self.index_current_waypoint += 1
            
            if self.index_current_waypoint >= len(self.path):
                self.finished_path = True
                self.index_current_waypoint = 0
                self.path = []
                self.path_grid = []
                return None
        
        return self.go_to_waypoint(current_position, current_orientation, odometer_values, lidar_values, ray_angles, found_and_near_wounded)

    def is_path_blocked(self, current_position, current_orientation, lidar_values, ray_angles):
        """
        Determine whether the path to the current waypoint is blocked by an obstacle using a cone of lidar_values in the direction of the current waypoint.
        """
        if self.finished_path or not self.path:
            return False

        current_waypoint = self.path[self.index_current_waypoint]
        direction_vector = current_waypoint - current_position
        dist_to_waypoint = np.linalg.norm(direction_vector)

        path_direction = np.arctan2(direction_vector[1], direction_vector[0])
        
        relative_angle = path_direction - current_orientation
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi     # Wrap the angle to [-pi, pi]
        
        angle_diffs = ray_angles - relative_angle
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        
        cone_mask = np.abs(angle_diffs) < self.path_params.OBSTACLE_CONE_ANGLE
        
        if not np.any(cone_mask):
            return False

        distances = lidar_values[cone_mask]
        min_dist = np.min(distances)
        
        if min_dist < dist_to_waypoint and min_dist < self.path_params.THRESHOLD_BLOCKED_PATH:
            return True
        
        return False
    
    def go_to_waypoint(self, current_position, current_orientation, odometer_values, lidar_values, ray_angles, found_and_near_wounded=False):
        command = {
            "forward": 0.0,
            "lateral": 0.0,
            "rotation": 0.0,
            "grasper": 1 if found_and_near_wounded else 0
        }

        if self.index_current_waypoint == 0:
            previous_waypoint = self.initial_point_path
        else:
            previous_waypoint = self.path[self.index_current_waypoint-1]

        current_waypoint = self.path[self.index_current_waypoint]
        
        direction_vector = current_waypoint - previous_waypoint
        if np.linalg.norm(direction_vector) > 0:
            direction_unit = direction_vector / np.linalg.norm(direction_vector)
        else:
            direction_unit = np.zeros(2)

        drone_vector = current_position - previous_waypoint # From previous waypoint to drone position
        
        # ANGLE CONTROL
        path_direction = np.arctan2(direction_vector[1], direction_vector[0])
        
        angle_error = path_direction - current_orientation
        
        command = self.rotation_pid.update_command(
            command, 
            angle_error, 
            odometer_values
        )

        # LATERAL CONTROL
        if np.linalg.norm(direction_vector) == 0:
            perpendicular_distance = 0.0
        else:
            perpendicular_distance = -np.cross(direction_unit, drone_vector) # Orthogonal distance from drone to the path segment

        perpendicular_error = perpendicular_distance + self.obstacle_avoidance_lateral_offset(lidar_values)

        if abs(angle_error) <= self.path_params.MAX_ANGLE_ERROR: # Allow lateral control only if the angle error is within a certain threshold
            command = self.lateral_pid.update_command(
                command,
                perpendicular_error, 
                odometer_values
            )
        
        # FORWARD CONTROL
        if np.linalg.norm(direction_vector) == 0:
            parallel_distance = 0.0
        else:
            parallel_distance = np.linalg.norm(direction_vector) - np.dot(direction_unit, drone_vector) # Remaining distance to waypoint along the path

        parallel_error = parallel_distance + self.obstacle_avoidance_longitudinal_offset(lidar_values)

        if abs(angle_error) <= self.path_params.MAX_ANGLE_ERROR: # Allow forward control only if the angle error is within a certain threshold
            command = self.forward_pid.update_command(
                command,
                parallel_error,
                odometer_values
            )
        
        return command
    
    def obstacle_avoidance_lateral_offset(self, lidar_values):
        """
        Calculates a lateral offset distance to avoid very close obstacles.
        """
        close_obstacle_mask = (lidar_values <= self.path_params.MAX_INFLATION_OBSTACLE)
        
        if not np.any(close_obstacle_mask):
            return 0.0

        relevant_lidar = lidar_values[close_obstacle_mask]
        relevant_offset_mask = self.lateral_offset_mask[close_obstacle_mask]

        # Calculate repulsion: closer obstacles create larger values.
        repulsion_magnitude = self.path_params.MAX_INFLATION_OBSTACLE - relevant_lidar
        
        weighted_repulsion = repulsion_magnitude * relevant_offset_mask
        
        total_offset = np.mean(weighted_repulsion)

        return total_offset
    
    def obstacle_avoidance_longitudinal_offset(self, lidar_values):
        """
        Calculates a forward/backward offset distance to avoid very close obstacles.
        """
        close_obstacle_mask = (lidar_values <= self.path_params.MAX_INFLATION_OBSTACLE)
        
        if not np.any(close_obstacle_mask):
            return 0.0

        relevant_lidar = lidar_values[close_obstacle_mask]
        relevant_offset_mask = self.longitudinal_offset_mask[close_obstacle_mask]

        # Calculate repulsion: closer obstacles create larger values.
        repulsion_magnitude = self.path_params.MAX_INFLATION_OBSTACLE - relevant_lidar
        
        weighted_repulsion = repulsion_magnitude * relevant_offset_mask
        
        total_offset = np.mean(weighted_repulsion)

        return total_offset