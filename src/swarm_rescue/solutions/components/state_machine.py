from enum import Enum, auto
from swarm_rescue.solutions.components.state_handlers import *

class DroneState(Enum):
        """
        All the states of the drone as a state machine
        """
        WAITING = auto()    # Assigns 1

        AVOIDING_OBSTACLE = auto()  # Assigns 2 etc ... This allows to easily add new states

        SEARCHING_WALL = auto()
        FOLLOWING_WALL = auto()

        CHOOSING_NEW_FRONTIER = auto()
        GOING_TO_FRONTIER = auto()

        GRASPING_WOUNDED = auto()
        SEARCHING_RESCUE_CENTER = auto()
        GOING_RESCUE_CENTER = auto()

        SEARCHING_RETURN_AREA = auto()
        GOING_RETURN_AREA = auto()

        EVALUATE_END_OF_MISSION = auto()
        CHOOSING_NEW_WOUNDED = auto()
        GOING_TO_WOUNDED = auto()
        END_OF_MISSION = auto()

class DroneStateMachine:
    def __init__(self, drone):
        self.drone = drone
        self.current_state = DroneState.WAITING
        self.previous_state = DroneState.WAITING
        
        # Initialize all state handlers
        self.states = {
            DroneState.WAITING: WaitingState(drone),
            DroneState.AVOIDING_OBSTACLE: AvoidingObstacleState(drone),
            DroneState.SEARCHING_WALL: SearchingWallState(drone),
            DroneState.FOLLOWING_WALL: FollowingWallState(drone),
            DroneState.GRASPING_WOUNDED: GraspingWoundedState(drone),
            DroneState.SEARCHING_RESCUE_CENTER: SearchingRescueCenterState(drone),
            DroneState.GOING_RESCUE_CENTER: GoingRescueCenterState(drone),
            DroneState.CHOOSING_NEW_FRONTIER: ChoosingNewFrontierState(drone),
            DroneState.GOING_TO_FRONTIER: GoingToFrontierState(drone),
            DroneState.EVALUATE_END_OF_MISSION: EvaluateEndOfMissionState(drone),
            DroneState.CHOOSING_NEW_WOUNDED: ChoosingNewWoundedState(drone),
            DroneState.GOING_TO_WOUNDED: GoingToWoundedState(drone),
            DroneState.END_OF_MISSION: EndOfMissionState(drone),
        }
        
        # Define state transitions
        self.transitions = {
            DroneState.WAITING: {
                "near_obstacle": DroneState.AVOIDING_OBSTACLE,
                "found_wounded": DroneState.GRASPING_WOUNDED,
                "waiting_time_over": DroneState.CHOOSING_NEW_FRONTIER
            },
            DroneState.AVOIDING_OBSTACLE: {
                "far_from_obstacle": DroneState.WAITING
            },
            DroneState.GRASPING_WOUNDED: {
                "lost_wounded": DroneState.WAITING,
                "holding_wounded": DroneState.SEARCHING_RESCUE_CENTER
            },
            DroneState.SEARCHING_RESCUE_CENTER: {
                "near_obstacle": DroneState.AVOIDING_OBSTACLE,
                "lost_rescue_center": DroneState.WAITING,
                "found_rescue_center": DroneState.GOING_RESCUE_CENTER,
                "is_path_blocked": DroneState.WAITING
            },
            DroneState.GOING_RESCUE_CENTER: {
                "near_obstacle": DroneState.AVOIDING_OBSTACLE,
                "lost_rescue_center": DroneState.WAITING,
                "is_too_close_to_rescue_center": DroneState.WAITING
            },
            DroneState.CHOOSING_NEW_FRONTIER: {
                "near_obstacle": DroneState.AVOIDING_OBSTACLE,
                "available_frontier": DroneState.GOING_TO_FRONTIER,
                "no_available_frontier": DroneState.EVALUATE_END_OF_MISSION
            },
            DroneState.GOING_TO_FRONTIER: {
                "near_obstacle": DroneState.AVOIDING_OBSTACLE,
                "finished_path": DroneState.CHOOSING_NEW_FRONTIER,
                "found_wounded": DroneState.GRASPING_WOUNDED,
                "is_near_rescuing_drone": DroneState.WAITING,
                "is_path_blocked": DroneState.WAITING
            },
            DroneState.EVALUATE_END_OF_MISSION: {
                "near_obstacle": DroneState.AVOIDING_OBSTACLE,
                "not_revisited_all_wounded_locations": DroneState.CHOOSING_NEW_WOUNDED,
                "sufficient_exploration_score": DroneState.END_OF_MISSION,
                "insufficient_exploration_score": DroneState.FOLLOWING_WALL
            },
            DroneState.CHOOSING_NEW_WOUNDED: {
                "near_obstacle": DroneState.AVOIDING_OBSTACLE,
                "revisited_all_wounded_locations": DroneState.EVALUATE_END_OF_MISSION,
                "not_revisited_all_wounded_locations": DroneState.GOING_TO_WOUNDED
            },
            DroneState.GOING_TO_WOUNDED: {
                "near_obstacle": DroneState.AVOIDING_OBSTACLE,
                "finished_path": DroneState.CHOOSING_NEW_WOUNDED,
                "found_wounded": DroneState.GRASPING_WOUNDED,
                "is_near_rescuing_drone": DroneState.WAITING,
                "is_path_blocked": DroneState.CHOOSING_NEW_WOUNDED
            },
            DroneState.END_OF_MISSION: {
                "near_obstacle": DroneState.AVOIDING_OBSTACLE,
            },
            DroneState.SEARCHING_WALL: {
                "found_wounded": DroneState.GRASPING_WOUNDED,
                "near_wall": DroneState.FOLLOWING_WALL,
                "is_near_rescuing_drone": DroneState.WAITING
            },
            DroneState.FOLLOWING_WALL: {
                "near_obstacle": DroneState.AVOIDING_OBSTACLE,
                "found_wounded": DroneState.GRASPING_WOUNDED,
                "lost_wall": DroneState.SEARCHING_WALL,
                "is_near_rescuing_drone": DroneState.WAITING
            }
        }
        
        self.active_handler = self.states[self.current_state]
    
    def update(self, conditions):
        """Update the state based on conditions"""
        self.previous_state = self.current_state
        
        # Check for state transitions
        for condition, next_state in self.transitions.get(self.current_state, {}).items():
            if conditions[condition]:
                self.current_state = next_state
                break
                
        # Handle state change
        if self.current_state != self.previous_state:
            self.states[self.previous_state].on_exit()
            self.active_handler = self.states[self.current_state]
            self.active_handler.on_enter()
            
        return self.current_state
        
    def handle_current_state(self):
        """Execute the behavior of the current state"""
        return self.active_handler.handle()