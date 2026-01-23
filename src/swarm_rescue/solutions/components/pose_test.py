from typing import Tuple, Optional, Dict
import numpy as np
from swarm_rescue.simulation.utils.utils import normalize_angle
from simulation.drone.drone_base import DroneBase
from solutions.utils.dataclasses_config import LocalizationParams
from solutions.components.pose import PoseEstimator
import pymunk
from simulation.utils.definitions import SPACE_DAMPING, SIMULATION_STEPS, ANGULAR_VELOCITY
from simulation.utils.constants import ANGULAR_SPEED_RATIO


class CommandPoseEstimator(PoseEstimator):
    """
    Command
    """
    def __init__(self, drone):
        self.drone = drone

        self.initialized = False
        self._position = np.zeros(2)
        self._angle = 0.0

        self.angular_ratio = ANGULAR_VELOCITY * ANGULAR_SPEED_RATIO

    def _lazy_init(self) -> None:
        """Initialize state from first valid GPS/compass readings."""
        self._position = self.drone.true_position().copy()
        self._angle = self.drone.true_angle()

        self.initialized = True
    
    @property
    def position(self) -> np.ndarray:
        return self._position.copy()
    
    @property
    def orientation(self) -> float:
        return self._angle
        
    def update(self,
               gps_position: Optional[np.ndarray] = None,
               compass_angle: Optional[float] = None,
               odometer_values: Optional[np.ndarray] = None,
               command: Optional[dict] = None,
               messages: Optional[list] = None) -> None:
        if not self.initialized:
            self._lazy_init()
            return
        
        angular_velocity = command["rotation"] * self.angular_ratio
        self._angle += angular_velocity