import arcade
from solutions.components.pose import Pose
from solutions.utils.dataclasses_config import VisualizationParams
from solutions.components.state_machine import DroneState

class VisualizationDrawer:
    def __init__(self, half_size_array, conv_grid_to_world):
        self.visualization_params = VisualizationParams()
        self._half_size_array = half_size_array
        self.conv_grid_to_world = conv_grid_to_world

    def draw_point(self, point, color=arcade.color.GO_GREEN):
        arcade.draw_circle_filled(point[0], point[1], 5, color)

    def draw_position(self, estimated_pose: Pose, ekf_pose):
        arcade.draw_circle_outline(
            estimated_pose.position[0] + self._half_size_array[0],
            estimated_pose.position[1] + self._half_size_array[1],
            10, arcade.color.RED
        )
        arcade.draw_circle_outline(
            ekf_pose.position[0] + self._half_size_array[0],
            ekf_pose.position[1] + self._half_size_array[1],
            10, arcade.color.BLUE
        )

    def draw_path(self, path):
        length = len(path)
        pt2 = None
        for ind_pt in range(length):
            pose = path[ind_pt]
            pt1 = pose + self._half_size_array
            if ind_pt > 0:
                arcade.draw_line(float(pt2[0]), float(pt2[1]), float(pt1[0]), float(pt1[1]), [125, 125, 125])
            pt2 = pt1

    def draw_top_layer(self, path, estimated_pose, ekf_pose, current_state: DroneState, next_frontier_centroid, next_frontier):
        if self.visualization_params.DRAW_PATH:
            self.draw_path(path)
        
        if self.visualization_params.DRAW_POSITION:
            self.draw_position(estimated_pose, ekf_pose)

        if current_state == DroneState.EXPLORING_FRONTIERS:
            if self.visualization_params.DRAW_FRONTIER_CENTROID and next_frontier_centroid is not None:
                self.draw_point(next_frontier_centroid + self._half_size_array)
            
            if self.visualization_params.DRAW_FRONTIER_POINTS and next_frontier is not None:
                for point in next_frontier.cells:
                    self.draw_point(self.conv_grid_to_world(*point) + self._half_size_array, color=arcade.color.AIR_FORCE_BLUE)