import numpy as np
from swarm_rescue.simulation.utils.utils import normalize_angle
from typing import List, Dict, Any, Optional

class PIDController:
    def __init__(self, 
                kp: float, 
                kd: float, 
                ki: float, 
                buffer_size: int = 10,
                mode: str = "generic"):
        """
        mode: "rotation", "lateral", or "forward"
        """
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.mode = mode
        self.error_history = [0.0] * buffer_size
        
    def compute(self, 
               error: float, 
               odometer_values: Optional[List[float]] = None) -> float:

        last_error = self.error_history[-1]
        self.error_history.pop(0)
        self.error_history.append(error)
        
        if odometer_values is not None:

            if self.mode == "rotation":
                error = normalize_angle(error)
                deriv_error = normalize_angle(odometer_values[2])

            elif self.mode == "lateral":
                deriv_error = -np.sin(odometer_values[1]) * odometer_values[0]

            elif self.mode == "forward":
                deriv_error = error - last_error

            else:
                deriv_error = error - last_error
    
        else:
            deriv_error = error - last_error
        
        proportional_correction = self.kp * error
        derivative_correction = self.kd * deriv_error
        integral_correction = self.ki * sum(self.error_history)

        correction = proportional_correction + derivative_correction + integral_correction
        
        return np.clip(correction, -1.0, 1.0)

    def get_command(self,
                     error: float, 
                     odometer_values: Optional[List[float]] = None) -> Dict[str, Any]:
        return self.compute(error, odometer_values)