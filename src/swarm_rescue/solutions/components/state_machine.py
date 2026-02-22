from enum import Enum, auto
from solutions.components.state_handlers import *

class DroneState(Enum):
        """
        All the states of the drone as a state machine
        """
        WAITING = auto()    # Assigns 1

        SEARCHING_WALL = auto()     # Assigns 2 etc ... This allows to easily add new states
        FOLLOWING_WALL = auto()

        EXPLORING_FRONTIERS = auto()

        GRASPING_WOUNDED = auto()
        SEARCHING_RESCUE_CENTER = auto()
        GOING_RESCUE_CENTER = auto()

        SEARCHING_RETURN_AREA = auto()
        GOING_RETURN_AREA = auto()

        EVALUATE_END_OF_MISSION = auto()
        END_OF_MISSION = auto()

class DroneStateMachine:
    def __init__(self, drone):
        self.drone = drone
        self.current_state = DroneState.WAITING
        self.previous_state = DroneState.WAITING
        
        # Initialize all state handlers
        self.states = {
            DroneState.WAITING: WaitingState(drone),
            DroneState.SEARCHING_WALL: SearchingWallState(drone),
            DroneState.FOLLOWING_WALL: FollowingWallState(drone),
            DroneState.GRASPING_WOUNDED: GraspingWoundedState(drone),
            DroneState.SEARCHING_RESCUE_CENTER: SearchingRescueCenterState(drone),
            DroneState.GOING_RESCUE_CENTER: GoingRescueCenterState(drone),
            DroneState.EXPLORING_FRONTIERS: ExploringFrontiersState(drone),
            DroneState.EVALUATE_END_OF_MISSION: EvaluateEndOfMissionState(drone),
            DroneState.END_OF_MISSION: EndOfMissionState(drone),
        }
        
        # Define state transitions
        self.transitions = {
            DroneState.WAITING: {
                "found_wounded": DroneState.GRASPING_WOUNDED,
                "waiting_time_over": DroneState.EXPLORING_FRONTIERS
            },
            DroneState.GRASPING_WOUNDED: {
                "lost_wounded": DroneState.WAITING,
                "holding_wounded": DroneState.SEARCHING_RESCUE_CENTER
            },
            DroneState.SEARCHING_RESCUE_CENTER: {
                "lost_rescue_center": DroneState.WAITING,
                "found_rescue_center": DroneState.GOING_RESCUE_CENTER
            },
            DroneState.GOING_RESCUE_CENTER: {
                "lost_rescue_center": DroneState.WAITING
            },
            DroneState.EXPLORING_FRONTIERS: {
                "found_wounded": DroneState.GRASPING_WOUNDED,
                "no_frontiers_left": DroneState.EVALUATE_END_OF_MISSION,
                "is_near_rescuing_drone": DroneState.WAITING
            },
            DroneState.EVALUATE_END_OF_MISSION: {
                "sufficient_exploration_score": DroneState.END_OF_MISSION,
                "insufficient_exploration_score": DroneState.FOLLOWING_WALL
            },
            DroneState.END_OF_MISSION: {
            },
            DroneState.SEARCHING_WALL: {
                "found_wounded": DroneState.GRASPING_WOUNDED,
                "near_obstacle": DroneState.FOLLOWING_WALL,
                "is_near_rescuing_drone": DroneState.WAITING
            },
            DroneState.FOLLOWING_WALL: {
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