import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from solutions.components.pose import PoseEstimatorManager

class PositionTracker:
    def __init__(self, pose_estimator_manager: PoseEstimatorManager):
        self.pose_estimator_manager = pose_estimator_manager
        self.all_estimator_names = list(pose_estimator_manager.estimators.keys())
        
        # Initialize data storage
        self.position_data: dict[str, list] = {name: [] for name in self.all_estimator_names}
        self.position_data["ground_truth"] = []

        self.performance_metrics: dict[str, float] = {}

    def update(self, true_position: np.ndarray) -> None:
        self.position_data["ground_truth"].append(true_position)
        
        # Record positions from different estimators
        for name, estimator in self.pose_estimator_manager.estimators.items():
            self.position_data[name].append(estimator.position)

    def _compute_errors(self, name: str) -> np.ndarray:
        ground_truth = np.array(self.position_data["ground_truth"])
        positions = np.array(self.position_data[name])
        return np.linalg.norm(positions - ground_truth, axis=1)

    def compute_performance_metrics(self) -> None:
        for name in self.all_estimator_names:
            errors = self._compute_errors(name)
            self.performance_metrics[name] = float(np.mean(errors**2))

    def export_performance_metrics(self, filepath: str = "solutions/utils/localization_metrics.csv"):
        self.compute_performance_metrics()

        df = pd.DataFrame(list(self.performance_metrics.items()), columns=["Method", "MSE"])
        df.to_csv(filepath, index=False)

        print(f"Performance metrics exported to {filepath}")

    def plot_errors(self) -> None:
        plt.figure(figsize=(10, 6))
        
        for name in self.all_estimator_names:
            errors = self._compute_errors(name)
            plt.plot(errors, label=name)
        
        plt.xlabel("Timestep")
        plt.ylabel("Position Error (m)")
        plt.title("Position Estimation Errors Over Time")
        plt.legend()
        plt.grid()
        plt.show()