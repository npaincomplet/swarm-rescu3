from typing import Optional
import numpy as np

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData

from solutions.components.pose import *
from solutions.components.astar import *
from solutions.components.communication import *
from solutions.components.grids import *
from solutions.components.path_planner import *
from solutions.utils.dataclasses_config import *
from solutions.components.pid_controller import *
from solutions.components.state_handlers import *
from solutions.components.state_machine import *
from solutions.components.path_controller import *
from solutions.components.sensor_manager import SensorManager
from solutions.components.health_manager import HealthManager
from solutions.components.visualization_drawer import VisualizationDrawer
from solutions.utils.data_serializer import PositionTracker

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
        self._init_misc()

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
        self.explored_all_frontiers = False
        self.next_frontier = None

        # Wall following
        self.wall_following_params = WallFollowingParams()

        # End of mission
        self.end_of_mission_params = EndOfMissionParams()

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
        self.initial_position = None
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

    def _init_misc(self):
        self.health_manager = HealthManager(self)
        self.last_command = self.null_command

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
        return self.sensor_manager.near_obstacle
        
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
    def is_near_rescue_center(self):
        return self.sensor_manager.is_near_rescue_center
        
    @property
    def min_dist_wounded(self):
        return self.sensor_manager.min_dist_wounded
    
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
    
    # Property to access exploration-related values
    
    @property
    def next_frontier_centroid(self):
        if isinstance(self.next_frontier, Frontier):
            return self.next_frontier.compute_centroid_pos()
        
        else:
            return None
    
    # Property misc

    @property
    def null_command(self):
        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
    
    @property
    def holding_wounded(self):
        return bool(self.grasper.grasped_wounded_persons)

    def reset_path_params(self):
        self.path_controller.reset_path()
    
    def get_sensor_conditions(self):
        is_near_rescuing_drone = self.check_near_rescuing_drone(threshold=30.0)

        if len(self.grid.frontiers) == 0:
            sufficient_exploration_score = self.compute_exploration_score() > self.end_of_mission_params.MIN_EXPLORATION_SCORE
        else:
            sufficient_exploration_score = False
        
        return {
            "near_obstacle": self.near_obstacle,
            "lost_wall": not self.near_obstacle,
            "found_wounded": self.found_wounded,
            "holding_wounded": self.holding_wounded,
            "lost_wounded": not self.found_wounded and not self.grasper.grasped_wounded_persons,
            "found_rescue_center": self.found_rescue_center,
            "lost_rescue_center": not self.grasper.grasped_wounded_persons,
            "no_frontiers_left": len(self.grid.frontiers) == 0,
            "waiting_time_over": self.step_waiting_count >= self.waiting_params.STEP_WAITING,
            "is_near_rescuing_drone": is_near_rescuing_drone,
            "sufficient_exploration_score": sufficient_exploration_score,
            "insufficient_exploration_score": not sufficient_exploration_score
        }

    def is_killed(self):
        in_kill_zone = self.lidar_values() is None

        return in_kill_zone
    
    def define_message_for_all(self):
        return self.communication_manager.prepare_outgoing_messages()

    def communication_management(self):
        self.communication_manager.process_incoming_messages()

    def control(self):
        if self.is_killed():
            # Drone in KillZone. Or at least no lidar available
            return self.null_command

        self.timestep_count += 1
        
        self.mapping(display=self.mapping_params.DISPLAY_MAP)
        self.communication_management()

        lidar_values = self.lidar_values()
        ray_angles = self.lidar_rays_angles()
        semantic_values = self.semantic_values()
        self.sensor_manager.process_sensors(lidar_values, ray_angles, 
                                            semantic_values,
                                            self.estimated_pose, 
                                            self.wounded_locked)

        # Update state machine with conditions
        conditions = self.get_sensor_conditions()
        self.state_machine.update(conditions)

        self.misc_management()

        self.draw_top_layer()

        # Execute current state behavior

        command = self.state_machine.handle_current_state() or self.null_command

        self.last_command = command

        return command

    def plan_path_to_rescue_center(self):
        start_pos = self.position
        target_pos = self.initial_position

        path = self.path_planner.plan_path_to_target(start_pos, target_pos, holds_wounded=True)
        self.path_controller.set_path(path, self.position)
    
    def assign_frontier(self):
        """
        Frontier assignment does not assume perfect information sharing between drones.
        Locally, each drone d1 is assigned to a frontier that would be optimal (in terms of task sharing) if each drone had the same information as d1.
        """
        frontiers = self.grid.update_frontiers()
        if not frontiers:
            return None

        # Collect known drone positions (self and others)
        drone_positions = [self.position]    # index 0 is self
        for _, pos in self.other_drones_pos:
            drone_positions.append(pos)

        num_drones = len(drone_positions)
        num_frontiers = len(frontiers)

        cost_matrix = np.zeros((num_drones, num_frontiers))
        for i, drone_pos in enumerate(drone_positions):
            for j, frontier in enumerate(frontiers):
                centroid = frontier.compute_centroid_pos()
                cost_matrix[i, j] = np.linalg.norm(drone_pos - centroid) / (frontier.size + 1)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)   # row_ind are drone indices in drone_positins and sorted, col_ind are frontier indices

        if 0 not in row_ind:    # index 0 is self. Case where self is not assigned a frontier
            best_frontier_index = np.argmin(cost_matrix[0, :])
            return frontiers[best_frontier_index]

        return frontiers[col_ind[0]]

    def plan_path_to_frontier(self):
        assigned_frontier = self.assign_frontier()
        
        if assigned_frontier is not None:
            self.next_frontier = assigned_frontier

            start_pos = self.position
            target_pos = self.next_frontier_centroid

            path = self.path_planner.plan_path_to_target(start_pos, target_pos, holds_wounded=False)
            if path is None:
                self.grid.delete_frontier_artifacts(self.next_frontier)
            else:
                self.path_controller.set_path(path, self.position)

        else:
            self.explored_all_frontiers = True

    def plan_path_to_return_area(self):
        start_pos = self.position
        target_pos = self.initial_position

        path = self.path_planner.plan_path_to_target(start_pos, target_pos, holds_wounded=False)
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
            found_and_near_wounded=found_and_near_wounded
        )
    
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

        if self.initial_position is None:
            self.initial_position = self.position
    
    def mapping(self, display = False):
        
        self.position_update()

        self.grid.update(pose=self.estimated_pose)
        
        if display and (self.timestep_count % 5 == 0):
             self.grid.display(self.grid.zoomed_grid,
                               self.estimated_pose,
                               title=f"Drone {self.identifier} zoomed occupancy grid")
    
    def compute_exploration_score(self):
        return self.grid.compute_exploration_score()

    def misc_management(self):
        self.health_manager.update()

    def logging_management(self):
        # true_position used only for logging purpose
        self.position_tracker.update(self.true_position())
        
        if self.timestep_count == 500:
            self.position_tracker.export_performance_metrics()

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