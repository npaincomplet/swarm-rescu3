from abc import ABC, abstractmethod
import numpy as np

class DroneState(ABC):
    """Abstract base class for all drone states"""
    
    def __init__(self, drone):
        self.drone = drone
    
    @abstractmethod
    def handle(self):
        """Execute state behavior and return command"""
        pass
    
    def on_enter(self):
        """Called when entering this state"""
        pass
    
    def on_exit(self):
        """Called when exiting this state"""
        pass


class WaitingState(DroneState):
    def on_enter(self):
        self.drone.step_waiting_count = 0

    def handle(self):
        self.drone.step_waiting_count += 1
        return self.drone.null_command

class AvoidingObstacleState(DroneState):
    def handle(self):
        lidar_values = self.drone.lidar_values()
        ray_angles = self.drone.lidar_rays_angles()
        
        min_idx = np.argmin(lidar_values)
        angle_min = ray_angles[min_idx]

        repulsion_speed = 1.0
        
        command = {
            "forward": -np.cos(angle_min) * repulsion_speed,
            "lateral": -np.sin(angle_min) * repulsion_speed,
            "rotation": 0.0, 
            "grasper": 0
        }
        return command


class SearchingWallState(DroneState):
    def handle(self):
        return {"forward": 0.5, "lateral": 0.0, "rotation": 0.0, "grasper": 0}


class FollowingWallState(DroneState):
    def handle(self):
        epsilon_wall_distance = self.drone.min_dist_wall - self.drone.wall_following_params.DIST_TO_STAY

        self.drone.logging_variables({
            "epsilon_wall_angle": self.drone.epsilon_wall_angle, 
            "epsilon_wall_distance": epsilon_wall_distance
        })

        command = {
            "forward": self.drone.wall_following_params.SPEED_FOLLOWING_WALL, 
            "lateral": 0.0, 
            "rotation": 0.0, 
            "grasper": 0
        }

        command["rotation"] = self.drone.rotation_pid.get_command(
            self.drone.epsilon_wall_angle, 
            self.drone.odometer_values()
        )
    
        command["lateral"] = self.drone.lateral_pid.get_command(
            epsilon_wall_distance, 
            self.drone.odometer_values()
        )

        return command


class GraspingWoundedState(DroneState):
    def handle(self):
        ready_to_grasp = self.drone.within_grasping_distance and self.drone.within_grasping_angle

        command = {
            "forward": self.drone.grasping_params.GRASPING_SPEED, 
            "lateral": 0.0, 
            "rotation": 0.0,
            "grasper": ready_to_grasp
        }

        command["rotation"] = self.drone.rotation_pid.get_command(
            self.drone.epsilon_wounded, 
            self.drone.odometer_values()
        )

        return command


class SearchingRescueCenterState(DroneState):
    def on_enter(self):
        self.drone.plan_path_to_rescue_center()
        
    def handle(self):
        return self.drone.follow_path(found_and_near_wounded=True)
    
    def on_exit(self):
        self.drone.reset_path_params()


class GoingRescueCenterState(DroneState):
    def handle(self):
        command = {
            "forward": 3 * self.drone.grasping_params.GRASPING_SPEED, 
            "lateral": 0.0, 
            "rotation": 0.0, 
            "grasper": 1
        }
        
        command["rotation"] = self.drone.rotation_pid.get_command(
            self.drone.epsilon_rescue_center, 
            self.drone.odometer_values()
        )

        return command


class ChoosingNewFrontierState(DroneState):
    def on_enter(self):
        self.drone.grid.update_frontiers()

    def handle(self):
        self.drone.plan_path_to_frontier()
        return self.drone.null_command


class GoingToFrontierState(DroneState):
    def handle(self):
        return self.drone.follow_path()


class EvaluateEndOfMissionState(DroneState):
    def handle(self):
        return self.drone.null_command


class ChoosingNewWoundedState(DroneState):
    def handle(self):
        self.drone.plan_path_to_next_sighting()
        return self.drone.null_command


class GoingToWoundedState(DroneState):
    def handle(self):
        return self.drone.follow_path()

    def on_exit(self):
        if self.drone.finished_path:
            self.drone.exploration_tracker.wounded_revisit_index += 1

class EndOfMissionState(DroneState):
    def on_enter(self):
        self.drone.plan_path_to_return_area()

    def handle(self):
        if self.drone.path_controller.finished_path and not self.drone.is_inside_return_area:
            self.drone.plan_path_to_return_area()

        if self.drone.path_controller.finished_path:
            return self.drone.null_command
        else:
            return self.drone.follow_path()