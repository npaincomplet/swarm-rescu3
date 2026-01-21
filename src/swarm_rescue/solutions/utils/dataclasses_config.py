from dataclasses import dataclass
import math

@dataclass
class MappingParams:
    RESOLUTION: int = 8     # 8 to 1 factor from simulation pixels to grid (efficiency)
    DISPLAY_MAP: bool = True
    DISPLAY_BINARY_MAP = True

@dataclass
class CommunicationParams:
    TIME_INTERVAL : int = 30
    MAX_INFO_DELAY: int = 2

@dataclass
class WaitingStateParams:
    STEP_WAITING: int = 20

@dataclass
class GraspingParams:
    GRASPING_SPEED: float = 0.3
    GRASPING_DIST: int = 30
    WOUNDED_CONFLICT_THRESHOLD: float = 20.0

@dataclass
class WallFollowingParams:
    DMAX: float = 60.0
    DIST_TO_STAY: float = 40.0
    SPEED_FOLLOWING_WALL: float = 0.3
    SPEED_TURNING: float = 0.05

@dataclass
class SensorParams:
    RESCUE_CENTER_DETECTION_THRESHOLD: float = 30.0
    NEAR_OBSTACLE_THRESHOLD: float = WallFollowingParams.DMAX # Beware of the circular dependency with WallFollowingParams

@dataclass
class PIDParams:
    KP_ANGLE: float = 9 / math.pi
    KD_ANGLE: float = KP_ANGLE / 10
    KI_ANGLE: float = 0.0
    
    KP_LATERAL: float = 0.3111
    KD_LATERAL: float = 1.3667
    KI_LATERAL: float = 0.0

    KP_FORWARD: float = 1.6
    KD_FORWARD: float = 11.0
    KI_FORWARD: float = 0.0

@dataclass
class PathControlParams:
    DISTANCE_CLOSE_WAYPOINT: int = 20
    SPEED_CLOSE_WAYPOINT: float = 2.0
    MAX_INFLATION_OBSTACLE: float = 20.0

    MAX_ANGLE_ERROR: float = math.pi / 15

@dataclass
class PhysicalParams:
    DRONE_RADIUS: float = 15.0

@dataclass
class PathPlanningParams:
    CONSTANT_COST: float = 1.0
    WORLD_INFLATION_RADIUS: float = PhysicalParams.DRONE_RADIUS * 3   # pixels
    GRID_INFLATION_RADIUS: int = int(WORLD_INFLATION_RADIUS / MappingParams.RESOLUTION)

    OBSTACLE_BASE_PENALTY: float = 10.0

    CAUTION_PENALTY_ALLOWANCE: float = 1.0
    SHORTCUT_PENALTY_ALLOWANCE: float = OBSTACLE_BASE_PENALTY / 4

@dataclass
class VisualisationParams:
    DRAW_POSITION: bool = True
    DRAW_PATH: bool = True
    DRAW_FRONTIER_CENTROID: bool = True
    DRAW_FRONTIER_POINTS: bool = True

@dataclass  # Relative to grids.py
class GridParams:
    OBSTACLE: int = 1
    FREE: int = 0
    UNDISCOVERED: int = -2

    MIN_FRONTIER_SIZE: int = 3
    CLUSTERING_EPSILON: float = 2.0

    EVERY_N: int = 3
    LIDAR_DIST_CLIP: float = 40.0
    MAX_RANGE_LIDAR_SENSOR_FACTOR: float = 0.9
    EMPTY_ZONE_VALUE: float = -0.602
    OBSTACLE_ZONE_VALUE: float = 2.0
    FREE_ZONE_VALUE: float = -4.0

    THRESHOLD_MIN: float = -40.0
    THRESHOLD_MAX: float = 40.0
    WORLD_BORDERS_VALUE: float = THRESHOLD_MAX
    FRONTIER_ARTIFACT_RESET_VALUE: float = THRESHOLD_MAX

    # Used for the ternary map conversion
    FREE_THRESHOLD: float = 0
    OBSTACLE_THRESHOLD: float = 0

@dataclass
class LocalizationParams:
    # AR1 noise
    GPS_NOISE_STD: float = 5.0
    COMPASS_NOISE_STD: float = 4.0 * math.pi / 180.0
    ALPHA_AR1: float = 0.98

    # Gaussian noise
    ODOMETER_DISTANCE_NOISE_STD: float = 0.2
    ODOMETER_ALPHA_NOISE_STD: float = 8.0 * math.pi / 180.0
    ODOMETER_THETA_NOISE_STD: float = 1.0 * math.pi / 180.0

@dataclass
class HealthParams:
    HEALTH_MEMORY_SIZE: int = 100

@dataclass
class BehaviourParams:
    try_not_couting_drone_as_obstacle: bool = True

@dataclass
class LogParams:
    RECORD_LOG: bool = False
    LOG_FILE: str = "logs/log.txt"
    LOG_INITIALIZED: bool = False
    FLUSH_INTERVAL: int = 50

    OUTPUT_DIR: str = "position_logs"
    SAVE_INTERVAL: int = 50
    GENERATE_PLOTS: bool = True