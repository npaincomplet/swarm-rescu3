from typing import Optional
import numpy as np
import arcade

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
        self.estimated_pose = Pose(size_area=self.size_area)
        self.odometer_pose = OdometerPose(self)
        self.kf_pose = KalmanFilterPose()
        self.augmented_pose = AugmentedEKFLocalization()
        self.test_pose = TestPose()
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
        self.next_frontier_centroid = None

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
        self.initial_position = np.zeros(2) # temporary value, will be set at first mapping
        self.path_control_params = PathControlParams()
        self.path_controller = PathController(
            self.path_control_params,
            self.rotation_pid,
            self.lateral_pid,
            self.forward_pid,
            self.lidar().ray_angles
        )

    def _init_logging(self):
        self.log_params = LogParams()   
        self.timestep_count = 0
        self.log_initialized = False
        self.log_buffer = []

        self.position_tracker = PositionTracker(
            drone=self, 
            loc_methods={
                "measured_gps": self.measured_gps_position,
                "kf": lambda: self.kf_pose.position,
                "augmented_ekf": lambda: self.augmented_pose.position,
                "test_pose": lambda: self.test_pose.position
            }
        )

    def _init_visualization(self):
        self.visualisation_params = VisualisationParams()

    def _init_misc(self):
        self.health_manager = HealthManager(self)

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
    def position(self):
        """World position"""
        return self.estimated_pose.position
    
    @property
    def orientation(self):
        return self.estimated_pose.orientation
    
    def reset_path_params(self):
        self.path_controller.reset_path()
    
    def get_sensor_conditions(self):
        is_near_rescuing_drone = self.check_near_rescuing_drone(threshold=30.0)
        
        return {
            "near_obstacle": self.near_obstacle,
            "lost_wall": not self.near_obstacle,
            "found_wounded": self.found_wounded,
            "holding_wounded": bool(self.grasper.grasped_wounded_persons),
            "lost_wounded": not self.found_wounded and not self.grasper.grasped_wounded_persons,
            "found_rescue_center": self.found_rescue_center,
            "lost_rescue_center": not self.grasper.grasped_wounded_persons,
            "no_frontiers_left": len(self.grid.frontiers) == 0,
            "waiting_time_over": self.step_waiting_count >= self.waiting_params.STEP_WAITING,
            "is_near_rescuing_drone": is_near_rescuing_drone
        }

    def is_killed(self):
        in_kill_zone = self.lidar().get_sensor_values() is None

        return in_kill_zone
    
    def define_message_for_all(self):
        return self.communication_manager.prepare_outgoing_messages()

    def communication_management(self):
        self.communication_manager.process_incoming_messages()

    def control(self):
        if self.is_killed():
            # Drone in KillZone. Or at least no lidar available
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        self.timestep_count += 1
        
        self.mapping(display=self.mapping_params.DISPLAY_MAP)
        self.communication_management()

        lidar_values = self.lidar().get_sensor_values()
        ray_angles = self.lidar().ray_angles
        semantic_values = self.semantic_values()
        self.sensor_manager.process_sensors(lidar_values, ray_angles, 
                                            semantic_values, 
                                            self.estimated_pose, 
                                            self.wounded_locked)

        # Update state machine with conditions
        conditions = self.get_sensor_conditions()
        self.state_machine.update(conditions)

        self.misc_management()

        # Reset waiting count if entering waiting state
        if (self.current_state == DroneState.WAITING and 
            self.previous_state != DroneState.WAITING):
            self.step_waiting_count = 0

        self.visualise_actions()

        # Execute current state behavior
        if self.timestep_count == 1:
            true_initial_state = np.array([self.true_position()[0], self.true_position()[1], self.true_angle(), 0,0,0])
            self.odometer_pose = OdometerPose(self)
            initial_state = np.array([self.measured_gps_position()[0],self.measured_gps_position()[1], self.measured_compass_angle(),0,0,0])
            self.augmented_pose = AugmentedEKFLocalization(initial_state=initial_state)
            self.test_pose = TestPose(initial_state=initial_state)

        command = self.state_machine.handle_current_state()

        ### Command should be computed after position is updated (as is done for estimated_pose in mapping method)

        self.odometer_pose.update(
            gps_position=self.measured_gps_position(),
            compass_angle=self.measured_compass_angle(),
            odometer_values=self.odometer_values()
        )

        self.kf_pose.update(
            gps_position=self.measured_gps_position(),
            compass_angle=self.measured_compass_angle(),
            odometer_values=self.odometer_values()
        )

        self.augmented_pose.step(
            odometer_values=self.odometer_values(),
            gps_position=self.measured_gps_position(),
            compass_angle=self.measured_compass_angle()
        )

        self.test_pose.step(
            odometer_values=self.odometer_values(),
            gps_position=self.measured_gps_position(),
            compass_angle=self.measured_compass_angle()
        )

        self.logging_management()

        return command

    def plan_path_to_rescue_center(self):
        start_pos = self.position
        target_pos = self.initial_position

        path = self.path_planner.plan_path_to_target(start_pos, target_pos)
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
        drone_positions = [self.estimated_pose.position]    # index 0 is self
        for _, pos in self.other_drones_pos:
            drone_positions.append(pos)

        num_drones = len(drone_positions)
        num_frontiers = len(frontiers)

        cost_matrix = np.zeros((num_drones, num_frontiers))
        for i, drone_pos in enumerate(drone_positions):
            for j, frontier in enumerate(frontiers):
                centroid = frontier.compute_centroid_pos()
                cost_matrix[i, j] = np.linalg.norm(drone_pos - centroid) / (frontier.size() + 1)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)   # row_ind are drone indices in drone_positins and sorted, col_ind are frontier indices

        if 0 not in row_ind:    # index 0 is self. Case where self is not assigned a frontier
            best_frontier_index = np.argmin(cost_matrix[0, :])
            return frontiers[best_frontier_index]

        return frontiers[col_ind[0]]

    def plan_path_to_frontier(self):
        assigned_frontier = self.assign_frontier()
        
        if assigned_frontier is not None:
            self.next_frontier_centroid = assigned_frontier.compute_centroid_pos()
            start_pos = self.position
            target_pos = self.next_frontier_centroid
            path = self.path_planner.plan_path_to_target(start_pos, target_pos)
            if path is None:
                print("Assigned frontier unreachable, deleting artifacts.")
                self.grid.delete_frontier_artifacts(self.next_frontier)
            else:
                self.path_controller.set_path(path, self.estimated_pose.position)

        else:
            self.explored_all_frontiers = True

    def check_near_rescuing_drone(self, threshold, messages=None):
        """
        Checks if any received broadcast message indicates a drone (other than self)
        is grasping a wounded and is closer than the given threshold.
        """

        for _,broadcast_loc in self.wounded_locked :
            distance = np.linalg.norm(np.array(self.estimated_pose.position) - np.array(broadcast_loc))
            if distance < threshold:
                print("Near a rescuing drone")
                return True
        return False

    def follow_path(self, found_and_near_wounded):
        return self.path_controller.follow_path(
            self.estimated_pose.position,
            self.estimated_pose.orientation,
            self.odometer_values(),
            self.lidar().get_sensor_values(),
            self.lidar().ray_angles,
            found_and_near_wounded
        )
    
    def position_update(self):
        self.estimated_pose.update(
            gps_position=self.measured_gps_position(),
            compass_angle=self.measured_compass_angle(),
            odometer_values=self.odometer_values()
        )
    
    def mapping(self, display = False):
        
        self.position_update()

        self.grid.update(pose=self.estimated_pose)
        
        if display and (self.timestep_count % 5 == 0):
             self.grid.display(self.grid.zoomed_grid,
                               self.estimated_pose,
                               title=f"Drone {self.identifier} zoomed occupancy grid")
        
        if self.timestep_count == 1: # first iterations
            print("Starting control")
            start_x, start_y = self.measured_gps_position() # never none ? 
            print(f"Initial position: {start_x}, {start_y}")
            self.initial_position = self.position

    def misc_management(self):
        self.health_manager.update()

    def logging_management(self):
        self.position_tracker.update()
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
        if not self.log_params.record_log:
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

    def draw_point(self,point, color=arcade.color.GO_GREEN):
        arcade.draw_circle_filled(point[0], point[1], 5, color)

    def draw_position(self):
        """
        arcade.draw_circle_outline(
            self.estimated_pose.position[0] + self._half_size_array[0],
            self.estimated_pose.position[1] + self._half_size_array[1],
            10, arcade.color.RED
        )

        arcade.draw_circle_outline(
            self.kf_pose.position[0] + self._half_size_array[0],
            self.kf_pose.position[1] + self._half_size_array[1],
            10, arcade.color.GREEN
        )

        arcade.draw_circle_outline(
            self.augmented_pose.position[0] + self._half_size_array[0],
            self.augmented_pose.position[1] + self._half_size_array[1],
            10, arcade.color.BLACK
        )
        """
        pass

    def draw_path(self, path):
        length = len(path)
        pt2 = None
        for ind_pt in range(length):
            pose = path[ind_pt]
            pt1 = pose + self._half_size_array
            if ind_pt > 0:
                arcade.draw_line(float(pt2[0]),
                                 float(pt2[1]),
                                 float(pt1[0]),
                                 float(pt1[1]), [125,125,125])
            pt2 = pt1

    def draw_top_layer(self):
        if self.visualisation_params.DRAW_PATH:
            self.draw_path(self.path_controller.path)
        
        if self.visualisation_params.DRAW_POSITION:
            self.draw_position()

        if self.current_state == DroneState.EXPLORING_FRONTIERS:
            if self.visualisation_params.DRAW_FRONTIER_CENTROID and self.next_frontier_centroid is not None:
                self.draw_point(self.next_frontier_centroid + self._half_size_array)     # frame of reference change
            
            if self.visualisation_params.DRAW_FRONTIER_POINTS and self.next_frontier is not None:
                for point in self.next_frontier.cells:
                    self.draw_point(self.grid._conv_grid_to_world(*point) + self._half_size_array, color=arcade.color.AIR_FORCE_BLUE)     # frame of reference change

    def visualise_actions(self):
        """
        It's mandatory to use draw_top_layer to draw anything on the interface
        """
        self.draw_top_layer()