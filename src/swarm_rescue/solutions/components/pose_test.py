from typing import Optional, Tuple
import numpy as np
import math
from swarm_rescue.simulation.utils.utils import normalize_angle
from solutions.components.pose import PoseEstimator
from solutions.utils.dataclasses_config import LocalizationParams
from simulation.utils.definitions import SPACE_DAMPING, PYMUNK_STEPS, ANGULAR_VELOCITY, LINEAR_FORCE
from simulation.utils.constants import ANGULAR_SPEED_RATIO, LINEAR_SPEED_RATIO

class TrueEstimator(PoseEstimator):
    """
    True
    """
    def __init__(self, drone):
        self.drone = drone

    @property
    def position(self) -> np.ndarray:
        return self.drone.true_position().copy()
    
    @property
    def orientation(self) -> float:
        return self.drone.true_angle()
        
    def update(self,
               gps_position: Optional[np.ndarray] = None,
               compass_angle: Optional[float] = None,
               odometer_values: Optional[np.ndarray] = None,
               command: Optional[dict] = None,
               lidar_values: Optional[np.ndarray] = None,
               messages: Optional[list] = None) -> None:
        pass

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
        self.linear_ratio = LINEAR_FORCE * LINEAR_SPEED_RATIO
        self._mass = 50 * 10

    def _lazy_init(self) -> None:
        """Initialize state from first valid GPS/compass readings."""
        self._position = self.drone.measured_gps_position().copy()
        self._angle = self.drone.measured_compass_angle()
        self._velocity = np.zeros(2)

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
               lidar_values: Optional[np.ndarray] = None,
               messages: Optional[list] = None) -> None:
        if not self.initialized:
            self._lazy_init()
            return
        
        # Angular setup
        angular_velocity = command["rotation"] * self.angular_ratio

        # Linear setup
        # Commands translate to Forces (F = command * ratio)
        cmd_forward = command["forward"]
        cmd_lateral = command["lateral"]
        sqr_norm = cmd_forward ** 2 + cmd_lateral ** 2
        if sqr_norm > 1.0:
            norm = math.sqrt(sqr_norm)
            cmd_forward = cmd_forward / norm
            cmd_lateral = cmd_lateral / norm

        forward_force = cmd_forward * self.linear_ratio
        lateral_force = cmd_lateral * self.linear_ratio

        dt = 1 / PYMUNK_STEPS
        damping_factor = SPACE_DAMPING ** dt

        for _ in range(PYMUNK_STEPS):
            # 1. Rotate local force to global frame using current angle
            # Calculate force first using the current orientation
            c, s = np.cos(self._angle), np.sin(self._angle)
            fx = forward_force * c - lateral_force * s
            fy = forward_force * s + lateral_force * c
            
            # Update Angle
            self._angle += angular_velocity * dt
            angular_velocity *= damping_factor

            # 2. Calculate Acceleration (a = F / m)
            ax = fx / self._mass
            ay = fy / self._mass
            
            # 3. Integrate Velocity (v += a * dt) with Damping
            self._velocity[0] += ax * dt
            self._velocity[0] *= damping_factor
            
            self._velocity[1] += ay * dt
            self._velocity[1] *= damping_factor

            # 4. Integrate Position (p += v * dt)
            self._position[0] += self._velocity[0] * dt
            self._position[1] += self._velocity[1] * dt

        self._angle = normalize_angle(self._angle)