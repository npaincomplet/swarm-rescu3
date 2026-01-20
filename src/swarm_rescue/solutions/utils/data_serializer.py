import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Callable, Optional

class PositionTracker:
    """
    Tracks the accuracy of different position estimation methods by plotting their instantaneous error to the ground truth.
    Extended for agent-driven optimization: compute and export performance metrics.
    """
    
    def __init__(self, drone, loc_methods: Dict[str, Callable]):
        """
        loc_methods: dictionary of localization methods to track, with method names as keys and functions returning positions as values.
        """
        self.drone = drone
        self.loc_methods = loc_methods.copy()
        
        # Initialize data storage
        self.position_data = {method: [] for method in self.loc_methods.keys()}
        self.position_data["ground_truth"] = []

        self.performance_metrics = {method: 0 for method in self.loc_methods.keys()}

        self.timesteps = []

    def update(self):
        self.timesteps.append(self.drone.timestep_count)
        
        self.position_data["ground_truth"].append(self.drone.true_position())
        
        # Record positions from different methods
        for method, func in self.loc_methods.items():
            self.position_data[method].append(func())

    def compute_performance_metrics(self):
        """Compute macroscopic performance indicators of localization methods."""
        ground_truth = np.array(self.position_data["ground_truth"])
        for method_name in self.loc_methods.keys():
            positions = np.array(self.position_data[method_name])
            errors = np.linalg.norm(positions - ground_truth, axis=1)
            mse = np.mean(errors**2)

            self.performance_metrics[method_name] = mse

    def export_performance_metrics(self, filepath: str = "solutions/utils/localization_metrics.csv"):
        """Export performance metrics to CSV."""
        self.compute_performance_metrics()
        df = pd.DataFrame(list(self.performance_metrics.items()), columns=["Method", "MSE"])
        df.to_csv(filepath, index=False)
        print(f"Performance metrics exported to {filepath}")

    def plot_errors(self):
        plt.figure(figsize=(10, 6))
        
        ground_truth = np.array(self.position_data["ground_truth"])
        
        for method_name in self.loc_methods.keys():
            positions = np.array(self.position_data[method_name])
            errors = np.linalg.norm(positions - ground_truth, axis=1)
            
            plt.plot(self.timesteps, errors, label=method_name)
        
        plt.xlabel("Timestep")
        plt.ylabel("Position Error (m)")
        plt.title("Position Estimation Errors Over Time")
        plt.legend()
        plt.grid()
        plt.show()