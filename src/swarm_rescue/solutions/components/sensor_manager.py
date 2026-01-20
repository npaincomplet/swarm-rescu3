import numpy as np
import math
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.utils.utils import circular_mean
from solutions.utils.dataclasses_config import SensorParams, GraspingParams, WallFollowingParams

class SensorManager:
    """Manages sensor data processing without direct drone dependency"""
    
    def __init__(self):
        self.sensor_params = SensorParams()
        self.grasping_params = GraspingParams()
        self.wall_following_params = WallFollowingParams()
        
        # Lidar sensor values
        self.near_obstacle = False
        self.epsilon_wall_angle = 0.0
        self.min_dist_wall = 1000.0
        
        # Semantic sensor values
        self.found_wounded = False
        self.found_rescue_center = False
        self.epsilon_wounded = 0.0
        self.epsilon_rescue_center = 0.0
        self.is_near_rescue_center = False
        self.min_dist_wounded = 1000.0
        
    def reset_semantic_sensor_values(self):
        self.found_wounded = False
        self.found_rescue_center = False
        self.epsilon_wounded = 0.0
        self.epsilon_rescue_center = 0.0
        self.is_near_rescue_center = False
        self.min_dist_wounded = 1000.0
    
    def process_lidar_sensor(self, lidar_values, ray_angles):
        """
        Args:
            lidar_values: Array of distance values from lidar
            ray_angles: Array of angles corresponding to lidar values
        """
        if lidar_values is None:
            return
            
        self.min_dist_wall = min(lidar_values)
        angle_nearest_obstacle = ray_angles[np.argmin(lidar_values)]

        self.near_obstacle = self.min_dist_wall < self.sensor_params.NEAR_OBSTACLE_THRESHOLD
        self.epsilon_wall_angle = angle_nearest_obstacle - np.pi/2
    
    def process_semantic_sensor(self, semantic_values, estimated_pose, wounded_locked):
        """
        Args:
            semantic_values: Array of semantic sensor readings
            estimated_pose: Current pose estimation
            wounded_locked: List of wounded entities locked by other drones
        """
        if semantic_values is None:
            return
            
        self.reset_semantic_sensor_values()

        angles_list = []
        scores = []

        for data in semantic_values:
            if (data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER):
                angles_list.append(data.angle)

                self.found_rescue_center = True
                if data.distance < self.sensor_params.RESCUE_CENTER_DETECTION_THRESHOLD:
                    self.is_near_rescue_center = True
            
            # If the wounded person detected is held by nobody
            elif (data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON 
                  and not data.grasped):
                self.found_wounded = True

                v = (data.angle * data.angle) + \
                    (data.distance * data.distance / 10 ** 5)
                scores.append((v, data.angle, data.distance))

        if angles_list:
            self.epsilon_rescue_center = circular_mean(np.array(angles_list))

        filtered_scores = []

        for score in scores:
            conflict = False
            for wnd_locked in wounded_locked:
                dx = score[2] * math.cos(score[1] + estimated_pose.orientation)
                dy = score[2] * math.sin(score[1] + estimated_pose.orientation)
                detection_position = np.array(estimated_pose.position) + np.array([dx, dy])
                
                if np.linalg.norm(detection_position - np.array(wnd_locked[1])) < self.grasping_params.WOUNDED_CONFLICT_THRESHOLD:
                    conflict = True
                    print("Conflict of wounded")
                    break
                    
            if not conflict:
                filtered_scores.append(score)

        if filtered_scores:
            best_score_tuple = min(filtered_scores, key=lambda x: x[0])
            self.epsilon_wounded = best_score_tuple[1]
            self.min_dist_wounded = best_score_tuple[2]

    def process_sensors(self, lidar_values, ray_angles, semantic_values, estimated_pose, wounded_locked):
        self.process_lidar_sensor(lidar_values, ray_angles)
        self.process_semantic_sensor(semantic_values, estimated_pose, wounded_locked)