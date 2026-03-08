from collections import deque
import numpy as np
from swarm_rescue.solutions.utils.dataclasses_config import MemoryParams

class History:
    def __init__(self, memory_size: int):
        self._history = deque(maxlen=memory_size)
        self._initial_value = None
    
    def update(self, value_copy):
        if self._initial_value is None:
            self._initial_value = value_copy
            
        self._history.append(value_copy)
    
    @property
    def initial_value(self):
        return self._initial_value
    
    @property
    def last_value(self):
        return self._history[-1] if self._history else None
    
class PositionHistory(History):
    def __init__(self):
        super().__init__(MemoryParams.POSITION_MEMORY_SIZE)
        self.is_moving_threshold = MemoryParams.IS_MOVING_THRESHOLD
    
    def update(self, position: np.ndarray):
        pos_copy = position.copy()
        super().update(pos_copy)

    def is_moving(self) -> bool:
        if len(self._history) < 2:
            return False
        
        vector_displacements = np.array([self._history[i+1] - self._history[i] for i in range(len(self._history) - 1)])
        distance_displacements = np.linalg.norm(vector_displacements, axis=1)
        return np.max(distance_displacements) > self.is_moving_threshold

class OrientationHistory(History):
    def __init__(self):
        super().__init__(MemoryParams.POSITION_MEMORY_SIZE)
        self.is_rotating_threshold = MemoryParams.IS_ROTATING_THRESHOLD
    
    def update(self, orientation: float):
        super().update(orientation)
    
    def is_rotating(self) -> bool:
        if len(self._history) < 2:
            return False
        
        angle_diffs = []
        for i in range(len(self._history) - 1):
            diff = abs(self._history[i+1] - self._history[i])
            diff = min(diff, 2 * np.pi - diff)
            angle_diffs.append(diff)
        return max(angle_diffs) > self.is_rotating_threshold

class HealthHistory(History):
    """
    Health is an integer. 
    For any type of collision (speed, object), drone looses 1 point of health.
    Health loss is capped at 1 point per second.
    Therefore, just_took_damage can evaluate to True at max once per second.
    """
    def __init__(self):
        super().__init__(MemoryParams.HEALTH_MEMORY_SIZE)

    def update(self, health_value: float):
        super().update(health_value)

    def just_took_damage(self) -> bool:
        if len(self._history) < 2:
            return False
        return self._history[-1] < self._history[-2]

class CommandHistory(History):
    def __init__(self):
        super().__init__(MemoryParams.COMMAND_MEMORY_SIZE)

    def update(self, command):
        super().update(command)

class DroneMemory:
    def __init__(self):
        self.position = PositionHistory()
        self.orientation = OrientationHistory()
        self.health = HealthHistory()
        self.command = CommandHistory()

    def update(self, current_pos: np.ndarray, current_orientation: float, current_health: float, command):
        self.position.update(current_pos)
        self.orientation.update(current_orientation)
        self.health.update(current_health)
        self.command.update(command)