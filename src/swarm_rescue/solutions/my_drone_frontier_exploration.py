from typing import Optional
import numpy as np

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData

from swarm_rescue.solutions.components.pose import *
from swarm_rescue.solutions.components.astar import *
from swarm_rescue.solutions.components.communication import *
from swarm_rescue.solutions.components.grids import *
from swarm_rescue.solutions.components.path_planner import *
from swarm_rescue.solutions.utils.dataclasses_config import *
from swarm_rescue.solutions.components.pid_controller import *
from swarm_rescue.solutions.components.state_handlers import *
from swarm_rescue.solutions.components.state_machine import *
from swarm_rescue.solutions.components.path_controller import *
from swarm_rescue.solutions.components.sensor_manager import SensorManager
from swarm_rescue.solutions.components.memory import DroneMemory
from swarm_rescue.solutions.components.visualization_drawer import VisualizationDrawer
from swarm_rescue.solutions.utils.data_serializer import PositionTracker
from swarm_rescue.solutions.components.exploration_tracker import ExplorationTracker

from scipy.optimize import linear_sum_assignment


class MyDroneFrontex(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         **kwargs)
        
        # Initialize all drone systems
        self._init_mapping()
        self._init_path_planning()
        self._init_communication()
        self._init_state_machine()
        self._init_state_params()
        self._init_controllers()
        self._init_sensors()
        self._init_path_following()
        self._init_logging()
        self._init_visualization()
        self._init_memory()
        self._init_exploration_tracking()

    def _init_mapping(self):
        self.mapping_params = MappingParams()

        self.pose_estimator_manager = PoseEstimatorManager()
        self.pose_estimator_manager.add("ekf", EKFPoseEstimator(),active = True)

        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                 resolution=self.mapping_params.RESOLUTION,
                                 lidar=self.lidar(), semantic=self.semantic())
    
    def _init_path_planning(self):
        self.path_planner = PathPlanner(self.grid)

    def _init_communication(self):
        self.communication_params = CommunicationParams()
        self.communication_manager = CommunicationManager(self)

    def _init_state_machine(self):
        self.state_machine = DroneStateMachine(self)

    def _init_state_params(self):
        # Waiting state
        self.waiting_params = WaitingStateParams()
        self.step_waiting_count = 0
        
        # Grasping state
        self.grasping_params = GraspingParams()
        
        # Frontier exploration
        self.exploring_frontiers_params = ExploringFrontiersParams()
        self.next_frontier = None

        # Wall following
        self.wall_following_params = WallFollowingParams()

        # End of mission
        self.end_of_mission_params = EndOfMissionParams()
        self.next_wounded_sighting = None

    def _init_sensors(self):
        self.sensor_manager = SensorManager()

    def _init_controllers(self):
        self.pid_params = PIDParams()
        
        self.rotation_pid = PIDController(
            self.pid_params.KP_ANGLE, 
            self.pid_params.KD_ANGLE, 
            self.pid_params.KI_ANGLE,
            mode="rotation"
        )
        
        self.lateral_pid = PIDController(
            self.pid_params.KP_LATERAL, 
            self.pid_params.KD_LATERAL, 
            self.pid_params.KI_LATERAL,
            mode="lateral"
        )

        self.forward_pid = PIDController(
            self.pid_params.KP_FORWARD, 
            self.pid_params.KD_FORWARD, 
            self.pid_params.KI_FORWARD,
            mode="forward"
        )

    def _init_path_following(self):
        self.path_control_params = PathControlParams()
        self.path_controller = PathController(
            self.path_control_params,
            self.rotation_pid,
            self.lateral_pid,
            self.forward_pid,
            self.lidar_rays_angles()
        )

    def _init_logging(self):
        self.log_params = LogParams()   
        self.timestep_count = 0
        self.log_initialized = False
        self.log_buffer = []

        self.position_tracker = PositionTracker(self.pose_estimator_manager)

    def _init_visualization(self):
        self.visualization_drawer = VisualizationDrawer(
            half_size_array=self._half_size_array,
            conv_grid_to_world=self.grid._conv_grid_to_world
        )
        self.visualization_params = VisualizationParams()

    def _init_memory(self):
        self.memory = DroneMemory()

    def _init_exploration_tracking(self):
        self.exploration_tracker = ExplorationTracker()

    # Properties to access communication-related values

    @property
    def wounded_locked(self):
        return self.communication_manager.wounded_locked
    
    @property
    def other_drones_pos(self):
        return self.communication_manager.other_drones_pos
    
    # Properties to access sensor-related values

    @property
    def near_obstacle(self):
        return self.sensor_manager.near_obstacle or self.just_took_damage

    @property
    def near_other_drone(self):
        return self.sensor_manager.near_other_drone

    @property
    def near_wall(self):
        return self.sensor_manager.near_wall
        
    @property
    def min_dist_wall(self):
        return self.sensor_manager.min_dist_wall
        
    @property
    def epsilon_wall_angle(self):
        return self.sensor_manager.epsilon_wall_angle
        
    @property
    def found_wounded(self):
        return self.sensor_manager.found_wounded
        
    @property
    def found_rescue_center(self):
        return self.sensor_manager.found_rescue_center
        
    @property
    def epsilon_wounded(self):
        return self.sensor_manager.epsilon_wounded
        
    @property
    def epsilon_rescue_center(self):
        return self.sensor_manager.epsilon_rescue_center
        
    @property
    def is_too_close_to_rescue_center(self):
        """Usually means that the drone can't drop the wounded person in the rescue center"""
        return self.sensor_manager.is_too_close_to_rescue_center
        
    @property
    def min_dist_wounded(self):
        return self.sensor_manager.min_dist_wounded

    @property
    def all_in_sight_wounded_pos(self):
        return self.sensor_manager.all_in_sight_wounded_pos

    @property
    def within_grasping_distance(self):
        return self.found_wounded and self.min_dist_wounded < self.grasping_params.GRASPING_DISTANCE
    
    @property
    def within_grasping_angle(self):
        return self.found_wounded and self.epsilon_wounded < self.grasping_params.GRASPING_ANGLE
    
    @property
    def is_gps_enabled(self):
        return self.measured_gps_position() is not None
    
    # Properties to access state machine-related values

    @property
    def current_state(self):
        return self.state_machine.current_state
    
    @property
    def previous_state(self):
        return self.state_machine.previous_state
    
    # Properties to access path controller-related values

    @property
    def path(self):
        return self.path_controller.path
    
    @property
    def finished_path(self):
        return self.path_controller.finished_path
    
    # Property to access cinetic values

    @property
    def active_pose_estimator(self):
        return self.pose_estimator_manager.active
    
    @property
    def estimated_pose(self):
        return self.active_pose_estimator
    
    @property
    def position(self):
        """World position"""
        return self.active_pose_estimator.position.copy()
    
    @property
    def orientation(self):
        return self.active_pose_estimator.orientation
    
    @property
    def is_moving(self):
        return self.memory.position.is_moving() or self.memory.orientation.is_rotating()
    
    # Property to access exploration-related values

    @property
    def frontiers(self):
        return list(self.grid.frontiers)

    @property
    def available_frontier(self) -> bool:
        """Might be stale as self.grid.frontiers is only updated in ChoosingNewFrontierState.on_enter()"""
        return len(self.grid.frontiers) != 0
    
    @property
    def next_frontier_centroid(self):
        if isinstance(self.next_frontier, Frontier):
            return self.next_frontier.compute_centroid_pos()
        
        else:
            return None
    
    @property
    def initial_position(self):
        return self.memory.position.initial_value
    
    @property
    def wounded_sighting_positions(self):
        return self.exploration_tracker.wounded_sighting_positions
    
    @property
    def needs_to_revisit_wounded(self):
        return self.identifier < self.end_of_mission_params.NUMBER_DRONES_REVISIT_WOUNDED

    @property
    def revisited_all_assigned_wounded_locations(self):
        """Successfully revisited all wounded locations"""
        if self.needs_to_revisit_wounded:
            return self.exploration_tracker.revisited_all_assigned_wounded_locations
        else:
            return True

    # Property misc

    @property
    def null_command(self):
        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
    
    @property
    def last_command(self):
        return self.memory.command.last_value or self.null_command
    
    @property
    def holding_wounded(self):
        return bool(self.grasper.grasped_wounded_persons)
    
    @property
    def just_took_damage(self):
        return self.memory.health.just_took_damage()

    def reset_path_params(self):
        self.path_controller.reset_path()
    
    def get_sensor_conditions(self):
        is_near_rescuing_drone = self.check_near_rescuing_drone(threshold=30.0)

        sufficient_exploration_score = False if self.available_frontier else self.compute_exploration_score() > self.end_of_mission_params.MIN_EXPLORATION_SCORE

        is_path_blocked = False if self.finished_path else self.path_controller.is_path_blocked(self.position, self.orientation, self.lidar_values(), self.lidar_rays_angles())

        return {
            "finished_path": self.finished_path,
            "is_path_blocked": is_path_blocked,
            "near_obstacle": self.near_obstacle,
            "far_from_obstacle": not self.near_obstacle,
            "near_wall": self.near_wall,
            "lost_wall": not self.near_wall,
            "found_wounded": self.found_wounded,
            "holding_wounded": self.holding_wounded,
            "lost_wounded": not self.found_wounded and not self.holding_wounded,
            "found_rescue_center": self.found_rescue_center,
            "is_too_close_to_rescue_center": self.is_too_close_to_rescue_center,
            "lost_rescue_center": not self.grasper.grasped_wounded_persons,
            "available_frontier": self.available_frontier,
            "no_available_frontier": not self.available_frontier,
            "waiting_time_over": self.step_waiting_count >= self.waiting_params.STEP_WAITING,
            "is_near_rescuing_drone": is_near_rescuing_drone,
            "just_took_damage": self.just_took_damage,
            "sufficient_exploration_score": sufficient_exploration_score,
            "insufficient_exploration_score": not sufficient_exploration_score,
            "revisited_all_assigned_wounded_locations": self.revisited_all_assigned_wounded_locations,
            "not_revisited_all_assigned_wounded_locations": not self.revisited_all_assigned_wounded_locations
        }

    def is_killed(self):
        in_kill_zone = self.lidar_values() is None

        return in_kill_zone
    
    def define_message_for_all(self):
        return self.communication_manager.prepare_outgoing_messages()

    def communication_management(self):
        self.communication_manager.process_incoming_messages()

    def sanitize_command(self, command):
        sanitized_command = {
            "forward": np.clip(command["forward"], -1.0, 1.0),
            "lateral": np.clip(command["lateral"], -1.0, 1.0),
            "rotation": np.clip(command["rotation"], -1.0, 1.0),
            "grasper": command["grasper"] if command["grasper"] is not None else 0
        }

        return sanitized_command

    def control(self):
        if self.is_killed():
            # Drone in KillZone. Or at least no lidar available
            return self.null_command

        self.timestep_count += 1

        lidar_values = self.lidar_values()
        ray_angles = self.lidar_rays_angles()
        semantic_values = self.semantic_values()
        self.sensor_manager.process_sensors(lidar_values, ray_angles, 
                                            semantic_values,
                                            self.estimated_pose, 
                                            self.wounded_locked)
        
        self.mapping()
        self.communication_management()
        self.exploration_tracker_management()

        # Update state machine with conditions
        conditions = self.get_sensor_conditions()
        self.state_machine.update(conditions)

        self.misc_management()

        # self.draw_top_layer()

        # Execute current state behavior

        command = self.state_machine.handle_current_state() or self.null_command
        command = self.sanitize_command(command)

        self.memory.update(self.position, self.orientation, self.drone_health, command)

        # print(f"Drone {self.identifier} - State: {self.current_state}")

        return command

    def plan_path_to_rescue_center(self):
        start_pos = self.position
        target_pos = self.initial_position

        path = self.path_planner.plan_path_to_target(start_pos, target_pos, holds_wounded=True)
        self.path_controller.set_path(path, self.position)
    
    def assign_frontier(self):
        """
        Frontiers are updated in ChoosingNewFrontierState (state_handlers.py).

        Frontier assignment does not assume perfect information sharing between drones.
        Locally, each drone d1 is assigned to a frontier that would be optimal (in terms of task sharing) if each drone had the same information as d1.
        """
        frontiers = self.frontiers
        if not frontiers:
            return None

        # Collect known drone positions (self and others)
        drone_positions = [self.position]    # index 0 is self
        for _, pos in self.other_drones_pos:
            drone_positions.append(pos)

        num_drones = len(drone_positions)
        
        # Geometric splitting of frontiers to ensure enough targets for all drones
        while len(frontiers) < num_drones:
            largest_frontier_idx = np.argmax([f.size for f in frontiers])
            largest_frontier = frontiers[largest_frontier_idx]

            new_frontiers = largest_frontier.split()
            
            # Splitting did not create any new frontier
            if len(new_frontiers) == 1:
                break
            
            frontiers.pop(largest_frontier_idx)
            frontiers.extend(new_frontiers)

        num_frontiers = len(frontiers)

        cost_matrix = np.zeros((num_drones, num_frontiers))
        for i, drone_pos in enumerate(drone_positions):
            for j, frontier in enumerate(frontiers):
                centroid = frontier.compute_centroid_pos()
                distance = np.linalg.norm(drone_pos - centroid)
                cost = distance / (frontier.size + 1)
                
                if self.grid.check_line_of_sight(drone_pos, centroid):
                    cost *= self.exploring_frontiers_params.LINE_OF_SIGHT_COST_MULTIPLIER
                
                cost_matrix[i, j] = cost    # Using A* distance as a cost would be optimal but too cost-intensive

        row_ind, col_ind = linear_sum_assignment(cost_matrix)   # row_ind are drone indices in drone_positins and sorted, col_ind are frontier indices

        if 0 not in row_ind:    # index 0 is self. Case where self is not assigned a frontier
            best_frontier_index = np.argmin(cost_matrix[0, :])
            return frontiers[best_frontier_index]

        return frontiers[col_ind[0]]

    def plan_path_to_frontier(self):
        self.next_frontier = self.assign_frontier()

        if self.next_frontier is not None:
            start_pos = self.position
            target_pos = self.next_frontier_centroid

            path = self.path_planner.plan_path_to_target(start_pos, target_pos, holds_wounded=False)
            if path is None:
                self.grid.delete_frontier_artifacts(self.next_frontier)
                self.path_controller.reset_path()
            else:
                self.path_controller.set_path(path, self.position)

    def plan_path_to_return_area(self):
        start_pos = self.position
        target_pos = self.initial_position

        path = self.path_planner.plan_path_to_target(start_pos, target_pos, holds_wounded=False)
        self.path_controller.set_path(path, self.position)

    def assign_wounded_sighting(self):
        return self.exploration_tracker.assign_wounded_sighting()

    def plan_path_to_next_sighting(self):
        self.next_wounded_sighting = self.assign_wounded_sighting()

        if self.next_wounded_sighting is not None:
            start_pos = self.position
            target_pos = self.next_wounded_sighting

            path = self.path_planner.plan_path_to_target(start_pos, target_pos, holds_wounded=False)
            if path is None:
                self.path_controller.reset_path()
            else:
                self.path_controller.set_path(path, self.position)

    def check_near_rescuing_drone(self, threshold, messages=None):
        """
        Checks if any received broadcast message indicates a drone (other than self)
        is grasping a wounded and is closer than the given threshold.
        """

        for _,broadcast_loc in self.wounded_locked :
            distance = np.linalg.norm(np.array(self.position) - np.array(broadcast_loc))
            if distance < threshold:
                return True
        return False

    def follow_path(self, found_and_near_wounded=False):
        return self.path_controller.follow_path(
            self.position,
            self.estimated_pose.orientation,
            self.odometer_values(),
            self.lidar_values(),
            self.lidar_rays_angles(),
            self.near_other_drone,
            found_and_near_wounded=found_and_near_wounded
        )
    
    def exploration_tracker_management(self):
        self.merge_wounded_sighting(self.all_in_sight_wounded_pos)
    
    def merge_wounded_sighting(self, received_sighting_positions):
        self.exploration_tracker.merge_wounded_sighting(received_sighting_positions)

    def position_update(self):
        self.pose_estimator_manager.update_all(
            gps_position=self.measured_gps_position(),
            compass_angle=self.measured_compass_angle(),
            odometer_values=self.odometer_values(),
            command=self.last_command,
            lidar_values=self.lidar_values(),
            holding_wounded=self.holding_wounded,
            messages=[]
        )
    
    def mapping(self):
        self.position_update()

        self.grid.update(pose=self.estimated_pose, gps_enabled=self.is_gps_enabled)
    
    def compute_exploration_score(self):
        return self.grid.compute_exploration_score()

    def misc_management(self):
        display_zoomed_grid = self.visualization_params.DISPLAY_ZOOMED_GRID
        if display_zoomed_grid and (self.timestep_count % 5 == 0):
            title=f"Drone {self.identifier} zoomed occupancy grid"
            self.grid.display(title)

    # Use this function only at one place in the control method. Not handled othewise.
    # params : variables_to_log : dict of variables to log with keys as variable names and values as variable values.
    def logging_variables(self, variables_to_log):
        """
        Buffers and logs variables to the log file when the buffer reaches the flush interval.

        :param variables_to_log: dict of variables to log with keys as variable names 
                                and values as variable values.
        """
        if not self.log_params.RECORD_LOG:
            return

        # Initialize the log buffer if not already done
        if not hasattr(self, "log_buffer"):
            self.log_buffer = []

        # Append the current variables to the buffer
        log_entry = {"Timestep": self.timestep_count, **variables_to_log}
        self.log_buffer.append(log_entry)

        # Write the buffer to file when it reaches the flush interval
        if len(self.log_buffer) >= self.log_params.FLUSH_INTERVAL:
            mode = "w" if not self.log_initialized else "a"
            with open(self.log_params.LOG_FILE, mode) as log_file:
                # Write the header if not initialized
                if not self.log_initialized:
                    headers = ",".join(log_entry.keys())
                    log_file.write(headers + "\n")
                    self.log_initialized = True

                # Write buffered entries
                for entry in self.log_buffer:
                    line = ",".join(map(str, entry.values()))
                    log_file.write(line + "\n")

            # Clear the buffer
            self.log_buffer.clear()

    def draw_top_layer(self):
        return self.visualization_drawer.draw_top_layer(
            self.path,
            self.pose_estimator_manager,
            self.current_state,
            self.next_frontier
        )