from collections import deque
import numpy as np
from solutions.utils.dataclasses_config import MemoryParams

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
    
    def update(self, position: np.ndarray):
        pos_copy = position.copy()
        super().update(pos_copy)

class HealthHistory(History):
    def __init__(self):
        super().__init__(MemoryParams.HEALTH_MEMORY_SIZE)

    def update(self, health_value: float):
        super().update(health_value)

    def has_lost_health(self) -> bool:
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
        self.health = HealthHistory()
        self.command = CommandHistory()

    def update(self, current_pos: np.ndarray, current_health: float, command):
        self.position.update(current_pos)
        self.health.update(current_health)
        self.command.update(command)