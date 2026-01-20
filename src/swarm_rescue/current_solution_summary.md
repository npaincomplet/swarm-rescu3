**Advice prompting blueprint :**

Context:
[Paste one or more from the following sections]

Question:
[One or two precise design questions]

Constraints:
[Optional: what you are not willing to change]

## Swarm-Rescue — My starting solution

---

I already participated in the same contest last year and want to improve my solution. Here's what my strategy looked like during last year's finals.

Each behavior of the drone (mapping and pose, communication, state machine, controllers, sensors, path following, logging) is implemented in its own module to enforce separation of concerns and decoupling.

---

## Mapping

My strategy uses an occupancy grid to map the environment through simple cumulative updates.

#### Key Components:
- **Grid Structure**: 2D grid with world-to-grid coordinate conversions (inverting y-axis). Resolution-based discretization; borders marked as obstacles.
- **Probabilistic Updates**:
  - **Free Space**: Ray-cast from drone pose along lidar rays (sampled every N points), marking cells as free up to a confidence distance (clipped lidar range minus noise threshold). Uses Bresenham line algorithm.
  - **Obstacles**: Mark endpoints of rays hitting obstacles (below threshold) as occupied.
  - Values clipped to min/max thresholds; grid resized for zoomed view.
- **Ternary Representation**: Converts grid to OBSTACLE (high values), FREE (low values), UNDISCOVERED (zero) for path planning.
- **Frontier Detection**:
  - Identify boundaries between FREE and UNDISCOVERED via grid differences (horizontal/vertical).
  - Cluster frontier cells using DBSCAN (epsilon-based, min size threshold).
- **Path Computation**:
  - A* search on ternary map with simple costmap (every free cell has the same cost and obstacle cell have infinite cost) and octile heuristic from start to target (or nearest free cell if blocked).
  - Simplify path: remove collinear points, ensure line-of-sight, apply Ramer-Douglas-Peucker (tolerance 0.5).
- **Utilities**: Delete frontier artifacts (reset to obstacle value), merge grids by averaging.

## Pose

My pose estimation uses complex Augmented EKF on GPS and odometry but doesn't make use of command.

This doesn't allow for optimal performance, furthermore in no GPS zones where the estimation relies only on odometry (really noisy).

Suboptimal pose estimation causes :

- Artifacts in the drone's mapping which lead to fake frontiers, unsafe path planning

- Reduced cinetic control accuracy along the path (this control uses pose of the drone)

When using no-noise sensors (which is not allowed during evaluation) instead of real sensors, the exploration is substantially better.

---

## Path following

My design relies on following a sequence of waypoints using PID controllers for precise control:

- **Waypoint Progression**: Advances to the next waypoint when close enough (distance and speed thresholds), resetting when path completes.
- **Rotation Control**: Aligns drone orientation with path direction via angle error and PID.
- **Lateral Control**: Maintains position perpendicular to path, adjusted by obstacle avoidance using lidar rays (offset mask shifts left/right based on obstacles).
- **Forward Control**: Drives toward waypoint along path, gated by low angle error.
- **Obstacle Avoidance**: Computes lateral offset from lidar data to inflate around obstacles, ensuring safe navigation.

This integrates path tracking with reactive avoidance for robust autonomous flight.

---

## General behaviour

My strategy employs frontier-based exploration with discrete, non-adaptive assignments: paths are planned as fixed sequences of waypoints followed via PID controllers (without pure pursuit or continuous updates), and frontiers are assigned statically using the Hungarian algorithm to share objectives between drones, without continous updates of the objectives.

There is no collision avoidance mechanism between drones.

---

## State machine

### Core Ideas of the Drone State Machine Design

A state machine orchestrates the drone's behavior by managing discrete states that dictates the drone's behaviour (e.g., WAITING, EXPLORING_FRONTIERS, GRASPING_WOUNDED) and transitioning between them based on real-time sensor conditions (e.g., "found_wounded", "near_obstacle").