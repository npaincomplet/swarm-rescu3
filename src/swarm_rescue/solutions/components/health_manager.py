from collections import deque
from solutions.utils.dataclasses_config import HealthParams

class HealthManager:
    def __init__(self, drone):
        self.drone = drone
        self.params = HealthParams()
        self.health_history = deque(maxlen=self.params.HEALTH_MEMORY_SIZE)
    
    def update(self):
        self.health_history.append(self.drone.drone_health)

    def has_lost_health(self):
        if len(self.health_history) < 2:
            return False
        return self.health_history[-1] < self.health_history[-2]