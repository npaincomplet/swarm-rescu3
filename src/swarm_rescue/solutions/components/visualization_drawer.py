import arcade
import hashlib
from swarm_rescue.solutions.components.pose import PoseEstimator, PoseEstimatorManager
from swarm_rescue.solutions.utils.dataclasses_config import VisualizationParams
from swarm_rescue.solutions.components.state_machine import DroneState
from swarm_rescue.solutions.components.grids import Frontier

class VisualizationDrawer:
    def __init__(self, half_size_array, conv_grid_to_world):
        self.visualization_params = VisualizationParams()
        self._half_size_array = half_size_array
        self.conv_grid_to_world = conv_grid_to_world

    @staticmethod
    def _string_to_hex_color(name: str) -> str:
        """Generate a deterministic hex color from a string."""
        hash_bytes = hashlib.md5(name.encode()).hexdigest()
        return f"#{hash_bytes[:6]}"

    def draw_point(self, point, color=arcade.color.GO_GREEN):
        arcade.draw_circle_filled(point[0], point[1], 5, color)

    def draw_pose_estimator(self, pose_estimator: PoseEstimator, color=arcade.color.BLUE):
        arcade.draw_circle_outline(
            pose_estimator.position[0] + self._half_size_array[0],
            pose_estimator.position[1] + self._half_size_array[1],
            10, color
        )

    def draw_all_pose_estimators(self, pose_estimator_manager: PoseEstimatorManager):
        for name, estimator in pose_estimator_manager.estimators.items():
            estimator_color = arcade.color_from_hex_string(self._string_to_hex_color(name))
            self.draw_pose_estimator(estimator, color=estimator_color)

    def draw_path(self, path):
        length = len(path)
        pt2 = None
        for ind_pt in range(length):
            pose = path[ind_pt]
            pt1 = pose + self._half_size_array
            if ind_pt > 0:
                arcade.draw_line(float(pt2[0]), float(pt2[1]), float(pt1[0]), float(pt1[1]), [125, 125, 125])
            pt2 = pt1

    def draw_top_layer(self, path, pose_estimator_manager: PoseEstimatorManager, current_state: DroneState, next_frontier: Frontier):
        if self.visualization_params.DRAW_PATH:
            self.draw_path(path)
        
        if self.visualization_params.DRAW_POSITION:
            self.draw_all_pose_estimators(pose_estimator_manager)

        if current_state == DroneState.GOING_TO_FRONTIER:
            if self.visualization_params.DRAW_FRONTIER_CENTROID and next_frontier is not None:
                self.draw_point(next_frontier.compute_centroid_pos() + self._half_size_array)
            
            if self.visualization_params.DRAW_FRONTIER_POINTS and next_frontier is not None:
                for point in next_frontier.positions:
                    self.draw_point(point + self._half_size_array, color=arcade.color.AIR_FORCE_BLUE)