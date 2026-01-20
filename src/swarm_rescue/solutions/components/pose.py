from typing import Tuple, Optional
import numpy as np
from spg_overlay.utils.utils import normalize_angle
from solutions.utils.dataclasses_config import *

# Physics engine imports
import pymunk
from solutions.utils.drone_base_twin import DroneBaseTwin
from spg.utils.definitions import SPACE_DAMPING, SIMULATION_STEPS, LINEAR_FORCE, ANGULAR_VELOCITY
from spg_overlay.utils.constants import LINEAR_SPEED_RATIO, ANGULAR_SPEED_RATIO

class Pose:
    """
    GPS
    """
    def __init__(self, size_area: Tuple[float, float]):
        self.position = np.zeros(2)
        self.orientation = 0.0
        self.size_area = size_area
        self.gps = True
        
    def update(self, 
              gps_position: Optional[np.ndarray] = None,
              compass_angle: Optional[float] = None,
              odometer_values: Optional[list] = None) -> None:

        # GPS and compass are available
        if gps_position is not None and compass_angle is not None:
            self.position = gps_position
            self.orientation = compass_angle
            self.gps = True
            return
            
        # Dead reckoning when GPS is unavailable
        self.gps = False
        if odometer_values:
            # Update orientation
            self.orientation += odometer_values[2]
            self.orientation = normalize_angle(self.orientation)
            
            # Update position using odometry
            self.position[0] += np.cos(normalize_angle(odometer_values[1] + self.orientation)) * odometer_values[0]
            self.position[1] += np.sin(normalize_angle(odometer_values[1] + self.orientation)) * odometer_values[0]
            
            # Constrain position to world boundaries
            x_limit = int(self.size_area[0] / 2)
            y_limit = int(self.size_area[1] / 2)
            self.position[0] = np.clip(self.position[0], -x_limit, x_limit)
            self.position[1] = np.clip(self.position[1], -y_limit, y_limit)

class CommandPose:
    """
    Command
    """
    def __init__(self, drone):
        self.position = drone.true_position()
        self.orientation = drone.true_angle()
        
        # Create physics space
        space = pymunk.Space()
        space.gravity = pymunk.Vec2d(0.0, 0.0)
        space.damping = SPACE_DAMPING
        self.space = space
        
        # Create drone physics body
        self.base = DroneBaseTwin()
        
        # Set initial position and orientation of the physics body
        self.base._pm_body.position = (self.position[0], self.position[1])
        self.base._pm_body.angle = self.orientation
        
        self.space.add(self.base._pm_body, *self.base._pm_shapes)

        # At time t we receive gps at time t and odometry between t-1 and t therefore we need to use t-1 command
        self._last_command = {'forward': 0.0, 'lateral': 0.0, 'rotation': 0.0, 'grasper': 0}
        
    def update(self, command: dict) -> None:
        """Update position based on commands and time delta"""
        # Apply commands
        self.base._apply_commands(self._last_command)
        self._last_command = command
        
        # Step physics simulation
        for _ in range(SIMULATION_STEPS):
                self.space.step(1.0 / SIMULATION_STEPS)
        
        # Get updated position and orientation
        self.position = self.base._pm_body.position
        self.orientation = self.base._pm_body.angle

        print("NEXT CommandPose POSITION AND ORIENTATION")
        print(self.base._pm_body.velocity, self.base._pm_body.angular_velocity)

class StateCommandPose:
    """
    State-based CommandPose
    State : [x, y, theta, vx, vy, vtheta]
    """
    def __init__(self, loc_params = LocalizationParams, initial_state: Optional[np.ndarray] = None):
        if initial_state is None:
            self.state = np.zeros(6, dtype=float)
        else:
            self.state = initial_state.astype(float)

        self.loc_params = loc_params

        self._init_twin()

        # At time t we receive gps at time t and odometry between t-1 and t therefore we need to use t-1 command
        self._last_command = {'forward': 0.0, 'lateral': 0.0, 'rotation': 0.0, 'grasper': 0}

    def _init_twin(self):
        # Create physics space
        space = pymunk.Space()
        space.gravity = pymunk.Vec2d(0.0, 0.0)
        space.damping = SPACE_DAMPING
        self.space = space
        
        # Create drone physics body
        self.base = DroneBaseTwin()
        
        # Set initial position and orientation of the physics body
        self.base._pm_body.position = (self.position[0], self.position[1])
        self.base._pm_body.angle = self.orientation
        
        self.space.add(self.base._pm_body, *self.base._pm_shapes)

    @property
    def position(self):
        return self.state[0:2].copy()

    @property
    def orientation(self):
        return self.state[2]

    def step(self,
               gps_position: np.ndarray = None,
               compass_angle: float = None,
               odometer_values: list = None,
               command = None) -> None:
        previous_position = self.position
        previous_orientation = self.orientation

        self.base._pm_body.velocity = tuple(self.state[3:5])
        self.base._pm_body.angular_velocity = self.state[5]

        self._simulate_physics_step(self._last_command)
        self._last_command = command

        #### THIS IS THE INTERESTING PART ####

        print("NEXT POSE (Physics engine)")
        print(self.base._pm_body.velocity, self.base._pm_body.angular_velocity)

        self.state[3] = self.state[0] - previous_position[0]
        self.state[4] = self.state[1] - previous_position[1]

        # !!!! _pm_body.angular_velocity doesn't rely on past values (it is directly set in _apply_commands) so we can use it directly
        # At t+1, the best estimate of angular velocity using command doesn't rely on time t best estimate using every other methods
        self.state[5] = self.base._pm_body.angular_velocity

        #### IT SHOULD HELP SEE WHAT TO PUT AS base._pm_body PARAMETERS IN A FUSION CLASS ####

        print("NEXT POSE (State-based)")
        print(self.state[3:5],self.state[5])

    def _simulate_physics_step(self, command):
        self.base._apply_commands(command)
        
        # Step physics simulation
        for _ in range(SIMULATION_STEPS):
                self.space.step(1.0 / SIMULATION_STEPS)
        
        # Get updated position and orientation
        self.state[0] = self.base._pm_body.position.x
        self.state[1] = self.base._pm_body.position.y
        self.state[2] = normalize_angle(self.base._pm_body.angle)

class OdometerPose:
    """
    Odometer
    """
    def __init__(self, drone):
        self.position = drone.true_position()
        self.orientation = drone.true_angle()

    def update(self, 
              gps_position: Optional[np.ndarray] = None,
              compass_angle: Optional[float] = None,
              odometer_values: Optional[list] = None) -> None:
            
        # Update orientation
        self.orientation += odometer_values[2]
        self.orientation = normalize_angle(self.orientation)
        
        # Update position using odometry
        self.position[0] += np.cos(normalize_angle(odometer_values[1] + self.orientation)) * odometer_values[0]
        self.position[1] += np.sin(normalize_angle(odometer_values[1] + self.orientation)) * odometer_values[0]

class KalmanFilterPose:
    """
    Kalman GPS + odometry
    State : [x, y, theta]
    """
    def __init__(self):
        self.state = np.zeros(3)
        
        self.loc_params = LocalizationParams()

        GPS_COMPASS_MEASUREMENTS_COV = np.diag([self.loc_params.GPS_NOISE_STD**2, self.loc_params.GPS_NOISE_STD**2, self.loc_params.COMPASS_NOISE_STD**2])
        self.R = GPS_COMPASS_MEASUREMENTS_COV
        
        ODOMETER_MEASUREMENTS_COV = np.diag([self.loc_params.ODOMETER_DISTANCE_NOISE_STD**2, self.loc_params.ODOMETER_ALPHA_NOISE_STD**2, self.loc_params.ODOMETER_THETA_NOISE_STD**2])
        self.Q = ODOMETER_MEASUREMENTS_COV
        
        self.P = self.R.copy() # Initial covariance matrix is gps covariance matrix because the first measurement is gps

    @property
    def position(self):
        return self.state[0:2].copy()

    @property
    def orientation(self):
        return self.state[2]
        
    def update(self, 
              gps_position: Optional[np.ndarray] = None,
              compass_angle: Optional[float] = None,
              odometer_values: Optional[list] = None) -> None:
        self._predict(odometer_values)
            
        self._update_measurement(gps_position, compass_angle)
        
        # Normalize the angle
        self.state[2] = normalize_angle(self.state[2])
    
    def _predict(self, odometer_values: list) -> None:
        distance = odometer_values[0]
        direction = odometer_values[1]
        rotation = odometer_values[2]
        
        current_theta = self.orientation
        
        self.state[2] += rotation
        
        move_angle = normalize_angle(direction + current_theta)
        self.state[0] += np.cos(move_angle) * distance
        self.state[1] += np.sin(move_angle) * distance
        
        self.P += self.Q
    
    def _update_measurement(self, gps_position: np.ndarray, compass_angle: float) -> None:
        # Measurement vector [x, y, theta]
        z = np.array([gps_position[0], gps_position[1], compass_angle])
        
        # Observation matrix (maps state to measurements)
        H = np.eye(3)  # Identity for direct state observation
        
        # Innovation (measurement residual)
        y = z - self.state
        y[2] = normalize_angle(y[2])  # Normalize angle difference
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state += K @ y
        
        # Update covariance
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

class EKFPose:
    """
    EKF pour fusion GPS + odométrie
    État : [x, y, theta]
    """
    def __init__(self, loc_params):
        self.state = np.zeros(3)  # état [x, y, theta]

        # Bruits
        self.loc_params = loc_params
        self.R = np.diag([
            loc_params.GPS_NOISE_STD**2,
            loc_params.GPS_NOISE_STD**2,
            loc_params.COMPASS_NOISE_STD**2
        ])
        self.Q = np.diag([
            loc_params.ODOMETER_DISTANCE_NOISE_STD**2,
            loc_params.ODOMETER_ALPHA_NOISE_STD**2,
            loc_params.ODOMETER_THETA_NOISE_STD**2
        ])

        # Matrice de covariance initiale (confiance GPS)
        self.P = self.R.copy()

    @property
    def position(self):
        return self.state[0:2].copy()

    @property
    def orientation(self):
        return self.state[2]

    def update(self,
               gps_position: np.ndarray = None,
               compass_angle: float = None,
               odometer_values: list = None) -> None:
        if odometer_values is not None:
            self._predict(odometer_values)

        if gps_position is not None and compass_angle is not None:
            self._update_measurement(gps_position, compass_angle)

        # Normaliser orientation
        self.state[2] = normalize_angle(self.state[2])

    def _predict(self, odometer_values: list) -> None:
        d, alpha, dtheta = odometer_values
        x, y, theta = self.state

        # Modèle de transition (non-linéaire)
        theta_new = normalize_angle(theta + dtheta)
        x_new = x + d * np.cos(theta + alpha)
        y_new = y + d * np.sin(theta + alpha)

        self.state = np.array([x_new, y_new, theta_new])

        # Jacobien F = ∂f/∂x
        F = np.eye(3)
        F[0, 2] = -d * np.sin(theta + alpha)  # ∂x_new/∂theta
        F[1, 2] = d * np.cos(theta + alpha)   # ∂y_new/∂theta

        # Jacobien G = ∂f/∂u (pour propager bruit odométrie)
        G = np.zeros((3, 3))
        G[0, 0] = np.cos(theta + alpha)      # ∂x/∂d
        G[0, 1] = -d * np.sin(theta + alpha) # ∂x/∂alpha
        G[1, 0] = np.sin(theta + alpha)      # ∂y/∂d
        G[1, 1] = d * np.cos(theta + alpha)  # ∂y/∂alpha
        G[2, 2] = 1.0                        # ∂theta/∂dtheta

        # Propagation de la covariance
        self.P = F @ self.P @ F.T + G @ self.Q @ G.T

    def _update_measurement(self, gps_position: np.ndarray, compass_angle: float) -> None:
        # Mesure
        z = np.array([gps_position[0], gps_position[1], compass_angle])

        # Fonction de mesure : h(x) = x
        H = np.eye(3)

        # Innovation
        y = z - self.state
        y[2] = normalize_angle(y[2])  # corriger écart d’angle

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Gain de Kalman
        K = self.P @ H.T @ np.linalg.inv(S)

        # Mise à jour état
        self.state = self.state + K @ y

        # Mise à jour covariance
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

class AugmentedEKFLocalization:
    """
    Augmented EKF for 2D pose with AR(1) GPS + compass noise.
    State vector (6):
        x = [px, py, theta, n_gx, n_gy, n_c]^T
    Where:
        - (px,py,theta) : drone pose
        - n_gx, n_gy     : additive AR(1) noise on GPS x & y measurements
        - n_c           : additive AR(1) noise on compass measurement

    Inputs:
        odom = [d, alpha, dtheta]  (distance, relative angle offset, orientation change)
        gps_position: np.array([gx, gy]) or None
        compass_angle: float or None

    Usage:
        lok = AugmentedEKFLocalization(LocalizationParams())
        lok.step(odometer_values=..., gps_position=..., compass_angle=...)
        pos = lok.position; theta = lok.orientation
    """

    def __init__(self, loc_params = LocalizationParams, initial_state: Optional[np.ndarray] = None):
        self.loc = loc_params
        # AR(1) parameter
        self.alpha_ar = float(self.loc.ALPHA_AR1)

        # State dimension
        self.nx = 6

        # Initialize state: default zeros (can be overridden)
        if initial_state is None:
            self.x = np.zeros(self.nx, dtype=float)
        else:
            assert initial_state.shape == (self.nx,)
            self.x = initial_state.astype(float)

        # Measurement covariance for the *observed* measurements (will be used in update)
        # These represent the *stationary* variances of GPS and compass measurement noise.
        self.R_meas = np.diag([
            self.loc.GPS_NOISE_STD ** 2,
            self.loc.GPS_NOISE_STD ** 2,
            self.loc.COMPASS_NOISE_STD ** 2
        ])

        # Odometer noise covariance (in odom-measurement space)
        Q_odom = np.diag([
            self.loc.ODOMETER_DISTANCE_NOISE_STD ** 2,
            self.loc.ODOMETER_ALPHA_NOISE_STD ** 2,
            self.loc.ODOMETER_THETA_NOISE_STD ** 2
        ])

        # AR(1) innovation variance q such that Var(n) = q / (1 - alpha^2) = sigma_n^2
        # => q = (1 - alpha^2) * sigma_n^2
        q_g = (1.0 - self.alpha_ar ** 2) * (self.loc.GPS_NOISE_STD ** 2)
        q_c = (1.0 - self.alpha_ar ** 2) * (self.loc.COMPASS_NOISE_STD ** 2)

        # We'll build the process noise for the augmented system inside predict()
        # but keep odom Q and AR innovations available
        self.Q_odom = Q_odom
        self.Q_ar_innov = np.diag([q_g, q_g, q_c])

        # Initial covariance: conservative, seed with some moderate uncertainty
        # Put larger uncertainty on pose if initial unknown.
        # Use measurement stationary variances for noise states (so P[3:6,3:6] ~= Var(n))
        P = np.zeros((self.nx, self.nx), dtype=float)
        pose_unc = max(1.0, self.loc.GPS_NOISE_STD)
        P[0, 0] = pose_unc ** 2
        P[1, 1] = pose_unc ** 2
        P[2, 2] = (2.0 * self.loc.COMPASS_NOISE_STD) ** 2
        P[3, 3] = self.loc.GPS_NOISE_STD ** 2   # var(n_gx)
        P[4, 4] = self.loc.GPS_NOISE_STD ** 2   # var(n_gy)
        P[5, 5] = self.loc.COMPASS_NOISE_STD ** 2
        self.P = P

        # Small regularization for numerical stability
        self._eps = 1e-9

    @property
    def position(self) -> np.ndarray:
        return self.x[0:2].copy()

    @property
    def orientation(self) -> float:
        return float(self.x[2])

    def step(self,
             odometer_values: Optional[Tuple[float, float, float]] = None,
             gps_position: Optional[np.ndarray] = None,
             compass_angle: Optional[float] = None) -> None:
        """
        Perform one EKF step: predict (if odom given) and update (if measurements given).
        """
        if odometer_values is not None:
            self.predict(odometer_values)

        # Build measurement vector and H matrix dynamically depending on available sensors
        # Our measurement function (complete) would be:
        #   h(x) = [px + n_gx, py + n_gy, theta + n_c]
        # so H_full = [[1,0,0,1,0,0],
        #              [0,1,0,0,1,0],
        #              [0,0,1,0,0,1]]
        H_full = np.zeros((3, self.nx))
        H_full[0, 0] = 1.0; H_full[0, 3] = 1.0
        H_full[1, 1] = 1.0; H_full[1, 4] = 1.0
        H_full[2, 2] = 1.0; H_full[2, 5] = 1.0

        z_list = []
        H_list = []
        R_list = []

        if gps_position is not None:
            # measurement entries for gps x and y
            z_list.append(gps_position[0])
            z_list.append(gps_position[1])
            H_list.append(H_full[0, :])
            H_list.append(H_full[1, :])
            R_list.append(self.R_meas[0, 0])
            R_list.append(self.R_meas[1, 1])

        if compass_angle is not None:
            z_list.append(compass_angle)
            H_list.append(H_full[2, :])
            R_list.append(self.R_meas[2, 2])

        if len(z_list) > 0:
            z = np.array(z_list, dtype=float)
            H = np.vstack(H_list)  # shape (m, 6)
            R = np.diag(R_list)    # shape (m, m)

            # Measurement prediction h(x)
            # For each measurement row, compute corresponding h
            # Use H to pick components: h = H @ x (works because measurement model linear in state)
            hx = H.dot(self.x)

            # For orientation residual, ensure angle difference normalized (if compass included)
            y = z - hx
            # find if a compass row exists and its index (compass was appended at last if present)
            # We'll normalize any angle residuals by checking rows where H has 1 at theta (index 2)
            for i in range(H.shape[0]):
                if abs(H[i, 2]) > 0.5:  # this row measures theta + n_c
                    y[i] = normalize_angle(y[i])

            S = H.dot(self.P).dot(H.T) + R
            # numerical stabilization
            S += np.eye(S.shape[0]) * self._eps

            K = self.P.dot(H.T).dot(np.linalg.inv(S))
            dx = K.dot(y)

            # Update state
            self.x = self.x + dx
            self.x[2] = normalize_angle(self.x[2])

            # Update covariance
            I = np.eye(self.nx)
            self.P = (I - K.dot(H)).dot(self.P)
            # ensure symmetrics.positio
            self.P = (self.P + self.P.T) / 2.0

    def predict(self, odometer_values: Tuple[float, float, float]) -> None:
        """
        EKF prediction step using odometry and AR(1) evolution of noise states.

        odometer_values: (d, alpha_rel, dtheta)
            - d: distance moved in last timestep
            - alpha_rel: relative heading of movement in the drone's reference frame
            - dtheta: orientation change in last step (body rotation)
        """

        d, alpha_rel, dtheta = odometer_values
        px, py, theta, n_gx, n_gy, n_c = self.x.copy()

        # Pose propagation
        theta_pred = normalize_angle(theta + dtheta)
        move_angle = normalize_angle(theta + alpha_rel)
        px_pred = px + d * np.cos(move_angle)
        py_pred = py + d * np.sin(move_angle)

        # AR(1) noise propagation: n' = alpha * n + w  (w ~ N(0, q_ar))
        n_gx_pred = self.alpha_ar * n_gx
        n_gy_pred = self.alpha_ar * n_gy
        n_c_pred  = self.alpha_ar * n_c

        x_pred = np.array([px_pred, py_pred, theta_pred, n_gx_pred, n_gy_pred, n_c_pred], dtype=float)

        # Build Jacobian F = df/dx (6x6)
        F = np.eye(self.nx)
        # derivatives of px_pred, py_pred wrt theta
        F[0, 2] = -d * np.sin(move_angle)  # d(px)/d(theta)
        F[1, 2] =  d * np.cos(move_angle)  # d(py)/d(theta)
        # derivatives of noise states: n' = alpha * n => partial wrt the previous n
        F[3, 3] = self.alpha_ar
        F[4, 4] = self.alpha_ar
        F[5, 5] = self.alpha_ar

        # Process noise: we consider two blocks of independent process noise:
        #   w_odom (3): odometry measurement noise (d, alpha_rel, dtheta) with cov Q_odom
        #   w_ar   (3): AR(1) innovations (w_gx, w_gy, w_c) with cov Q_ar_innov
        #
        # We build L matrix mapping process noise vector w = [w_odom (3), w_ar (3)] to state increments:
        # x' = f(x, u) + L @ w
        # L shape: 6 x 6
        L = np.zeros((self.nx, 6))

        # Partials of px_pred, py_pred, theta_pred wrt odom noises (d, alpha_rel, dtheta)
        # ∂px/∂d = cos(move_angle)
        L[0, 0] = np.cos(move_angle)
        # ∂px/∂alpha_rel = -d * sin(move_angle)
        L[0, 1] = -d * np.sin(move_angle)
        # ∂px/∂dtheta = 0 (we used theta + dtheta only in theta_pred; px depends on theta through move_angle which uses theta, not dtheta)
        L[0, 2] = 0.0

        # ∂py/∂d = sin(move_angle)
        L[1, 0] = np.sin(move_angle)
        # ∂py/∂alpha_rel = d * cos(move_angle)
        L[1, 1] = d * np.cos(move_angle)
        L[1, 2] = 0.0

        # ∂theta/∂dtheta
        L[2, 2] = 1.0

        # AR(1) innovations directly add to the noise states:
        # n_gx' = alpha * n_gx + w_gx => ∂n_gx'/∂w_gx = 1
        L[3, 3] = 1.0
        L[4, 4] = 1.0
        L[5, 5] = 1.0

        # Build process noise covariance (6x6), block diagonal
        Q_process = np.zeros((6, 6))
        Q_process[0:3, 0:3] = self.Q_odom
        Q_process[3:6, 3:6] = self.Q_ar_innov

        # Propagate covariance: P = F P F^T + L Q_process L^T
        self.P = F.dot(self.P).dot(F.T) + L.dot(Q_process).dot(L.T)

        # Assign predicted state
        self.x = x_pred
        self.x[2] = normalize_angle(self.x[2])

        # numerical stabilization
        self.P += np.eye(self.nx) * self._eps
        self.P = (self.P + self.P.T) / 2.0

    def set_pose(self, px: float, py: float, theta: float, gps_noise_estimates: Optional[Tuple[float, float, float]] = None):
        """
        Reset/initialize the pose portion of the filter (convenience).
        gps_noise_estimates optionally: (n_gx, n_gy, n_c) initial AR1 noise guesses.
        """
        self.x[0] = px
        self.x[1] = py
        self.x[2] = normalize_angle(theta)
        if gps_noise_estimates is not None:
            self.x[3] = gps_noise_estimates[0]
            self.x[4] = gps_noise_estimates[1]
            self.x[5] = gps_noise_estimates[2]

    def get_full_state(self) -> np.ndarray:
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()

class TestPose:
    """
    Augmented EKF for 2D pose with AR(1) GPS + compass noise.
    State vector (6):
        x = [px, py, theta, n_gx, n_gy, n_c]^T
    Where:
        - (px,py,theta) : drone pose
        - n_gx, n_gy     : additive AR(1) noise on GPS x & y measurements
        - n_c           : additive AR(1) noise on compass measurement

    Inputs:
        odom = [d, alpha, dtheta]  (distance, relative angle offset, orientation change)
        gps_position: np.array([gx, gy]) or None
        compass_angle: float or None

    Usage:
        lok = AugmentedEKFLocalization(LocalizationParams())
        lok.step(odometer_values=..., gps_position=..., compass_angle=...)
        pos = lok.position; theta = lok.orientation
    """

    def __init__(self, loc_params = LocalizationParams, initial_state: Optional[np.ndarray] = None):
        self.loc = loc_params
        # AR(1) parameter
        self.alpha_ar = float(self.loc.ALPHA_AR1)

        # State dimension
        self.nx = 6

        # Initialize state: default zeros (can be overridden)
        if initial_state is None:
            self.x = np.zeros(self.nx, dtype=float)
        else:
            assert initial_state.shape == (self.nx,)
            self.x = initial_state.astype(float)

        # Measurement covariance for the *observed* measurements (will be used in update)
        # These represent the *stationary* variances of GPS and compass measurement noise.
        self.R_meas = np.diag([1e-3, 1e-3, 1e-5])

        # Odometer noise covariance (in odom-measurement space)
        Q_odom = np.diag([
            self.loc.ODOMETER_DISTANCE_NOISE_STD ** 2,
            self.loc.ODOMETER_ALPHA_NOISE_STD ** 2,
            self.loc.ODOMETER_THETA_NOISE_STD ** 2
        ])

        # AR(1) innovation variance q such that Var(n) = q / (1 - alpha^2) = sigma_n^2
        # => q = (1 - alpha^2) * sigma_n^2
        q_g = (1.0 - self.alpha_ar ** 2) * (self.loc.GPS_NOISE_STD ** 2)
        q_c = (1.0 - self.alpha_ar ** 2) * (self.loc.COMPASS_NOISE_STD ** 2)

        # We'll build the process noise for the augmented system inside predict()
        # but keep odom Q and AR innovations available
        self.Q_odom = Q_odom
        self.Q_ar_innov = np.diag([q_g, q_g, q_c])

        # Initial covariance: conservative, seed with some moderate uncertainty
        # Put larger uncertainty on pose if initial unknown.
        # Use measurement stationary variances for noise states (so P[3:6,3:6] ~= Var(n))
        P = np.zeros((self.nx, self.nx), dtype=float)
        pose_unc = max(1.0, self.loc.GPS_NOISE_STD)
        P[0, 0] = pose_unc ** 2
        P[1, 1] = pose_unc ** 2
        P[2, 2] = (2.0 * self.loc.COMPASS_NOISE_STD) ** 2
        P[3, 3] = self.loc.GPS_NOISE_STD ** 2   # var(n_gx)
        P[4, 4] = self.loc.GPS_NOISE_STD ** 2   # var(n_gy)
        P[5, 5] = self.loc.COMPASS_NOISE_STD ** 2
        self.P = P

        # Small regularization for numerical stability
        self._eps = 1e-9

    @property
    def position(self) -> np.ndarray:
        return self.x[0:2].copy()

    @property
    def orientation(self) -> float:
        return float(self.x[2])

    def step(self,
             odometer_values: Optional[Tuple[float, float, float]] = None,
             gps_position: Optional[np.ndarray] = None,
             compass_angle: Optional[float] = None) -> None:
        """
        Perform one EKF step: predict (if odom given) and update (if measurements given).
        """
        if odometer_values is not None:
            self.predict(odometer_values)

        # Build measurement vector and H matrix dynamically depending on available sensors
        # Our measurement function (complete) would be:
        #   h(x) = [px + n_gx, py + n_gy, theta + n_c]
        # so H_full = [[1,0,0,1,0,0],
        #              [0,1,0,0,1,0],
        #              [0,0,1,0,0,1]]
        H_full = np.zeros((3, self.nx))
        H_full[0, 0] = 1.0; H_full[0, 3] = 1.0
        H_full[1, 1] = 1.0; H_full[1, 4] = 1.0
        H_full[2, 2] = 1.0; H_full[2, 5] = 1.0

        z_list = []
        H_list = []
        R_list = []

        if gps_position is not None:
            # measurement entries for gps x and y
            z_list.append(gps_position[0])
            z_list.append(gps_position[1])
            H_list.append(H_full[0, :])
            H_list.append(H_full[1, :])
            R_list.append(self.R_meas[0, 0])
            R_list.append(self.R_meas[1, 1])

        if compass_angle is not None:
            z_list.append(compass_angle)
            H_list.append(H_full[2, :])
            R_list.append(self.R_meas[2, 2])

        if len(z_list) > 0:
            z = np.array(z_list, dtype=float)
            H = np.vstack(H_list)  # shape (m, 6)
            R = np.diag(R_list)    # shape (m, m)

            # Measurement prediction h(x)
            # For each measurement row, compute corresponding h
            # Use H to pick components: h = H @ x (works because measurement model linear in state)
            hx = H.dot(self.x)

            # For orientation residual, ensure angle difference normalized (if compass included)
            y = z - hx
            # find if a compass row exists and its index (compass was appended at last if present)
            # We'll normalize any angle residuals by checking rows where H has 1 at theta (index 2)
            for i in range(H.shape[0]):
                if abs(H[i, 2]) > 0.5:  # this row measures theta + n_c
                    y[i] = normalize_angle(y[i])

            S = H.dot(self.P).dot(H.T) + R
            # numerical stabilization
            S += np.eye(S.shape[0]) * self._eps

            K = self.P.dot(H.T).dot(np.linalg.inv(S))
            dx = K.dot(y)

            # Update state
            self.x = self.x + dx
            self.x[2] = normalize_angle(self.x[2])

            # Update covariance
            I = np.eye(self.nx)
            self.P = (I - K.dot(H)).dot(self.P)
            # ensure symmetric
            self.P = (self.P + self.P.T) / 2.0

    def predict(self, odometer_values: Tuple[float, float, float]) -> None:
        """
        EKF prediction step using odometry and AR(1) evolution of noise states.

        odometer_values: (d, alpha_rel, dtheta)
            - d: distance moved in last timestep
            - alpha_rel: relative heading of movement in the drone's reference frame
            - dtheta: orientation change in last step (body rotation)
        """

        d, alpha_rel, dtheta = odometer_values
        px, py, theta, n_gx, n_gy, n_c = self.x.copy()

        # Pose propagation
        theta_pred = normalize_angle(theta + dtheta)
        move_angle = normalize_angle(theta + alpha_rel)
        px_pred = px + d * np.cos(move_angle)
        py_pred = py + d * np.sin(move_angle)

        # AR(1) noise propagation: n' = alpha * n + w  (w ~ N(0, q_ar))
        n_gx_pred = self.alpha_ar * n_gx
        n_gy_pred = self.alpha_ar * n_gy
        n_c_pred  = self.alpha_ar * n_c

        x_pred = np.array([px_pred, py_pred, theta_pred, n_gx_pred, n_gy_pred, n_c_pred], dtype=float)

        # Build Jacobian F = df/dx (6x6)
        F = np.eye(self.nx)
        # derivatives of px_pred, py_pred wrt theta
        F[0, 2] = -d * np.sin(move_angle)  # d(px)/d(theta)
        F[1, 2] =  d * np.cos(move_angle)  # d(py)/d(theta)
        # derivatives of noise states: n' = alpha * n => partial wrt the previous n
        F[3, 3] = self.alpha_ar
        F[4, 4] = self.alpha_ar
        F[5, 5] = self.alpha_ar

        # Process noise: we consider two blocks of independent process noise:
        #   w_odom (3): odometry measurement noise (d, alpha_rel, dtheta) with cov Q_odom
        #   w_ar   (3): AR(1) innovations (w_gx, w_gy, w_c) with cov Q_ar_innov
        #
        # We build L matrix mapping process noise vector w = [w_odom (3), w_ar (3)] to state increments:
        # x' = f(x, u) + L @ w
        # L shape: 6 x 6
        L = np.zeros((self.nx, 6))

        # Partials of px_pred, py_pred, theta_pred wrt odom noises (d, alpha_rel, dtheta)
        # ∂px/∂d = cos(move_angle)
        L[0, 0] = np.cos(move_angle)
        # ∂px/∂alpha_rel = -d * sin(move_angle)
        L[0, 1] = -d * np.sin(move_angle)
        # ∂px/∂dtheta = 0 (we used theta + dtheta only in theta_pred; px depends on theta through move_angle which uses theta, not dtheta)
        L[0, 2] = 0.0

        # ∂py/∂d = sin(move_angle)
        L[1, 0] = np.sin(move_angle)
        # ∂py/∂alpha_rel = d * cos(move_angle)
        L[1, 1] = d * np.cos(move_angle)
        L[1, 2] = 0.0

        # ∂theta/∂dtheta
        L[2, 2] = 1.0

        # AR(1) innovations directly add to the noise states:
        # n_gx' = alpha * n_gx + w_gx => ∂n_gx'/∂w_gx = 1
        L[3, 3] = 1.0
        L[4, 4] = 1.0
        L[5, 5] = 1.0

        # Build process noise covariance (6x6), block diagonal
        Q_process = np.zeros((6, 6))
        Q_process[0:3, 0:3] = self.Q_odom
        Q_process[3:6, 3:6] = self.Q_ar_innov

        # Propagate covariance: P = F P F^T + L Q_process L^T
        self.P = F.dot(self.P).dot(F.T) + L.dot(Q_process).dot(L.T)

        # Assign predicted state
        self.x = x_pred
        self.x[2] = normalize_angle(self.x[2])

        # numerical stabilization
        self.P += np.eye(self.nx) * self._eps
        self.P = (self.P + self.P.T) / 2.0

    def set_pose(self, px: float, py: float, theta: float, gps_noise_estimates: Optional[Tuple[float, float, float]] = None):
        """
        Reset/initialize the pose portion of the filter (convenience).
        gps_noise_estimates optionally: (n_gx, n_gy, n_c) initial AR1 noise guesses.
        """
        self.x[0] = px
        self.x[1] = py
        self.x[2] = normalize_angle(theta)
        if gps_noise_estimates is not None:
            self.x[3] = gps_noise_estimates[0]
            self.x[4] = gps_noise_estimates[1]
            self.x[5] = gps_noise_estimates[2]

    def get_full_state(self) -> np.ndarray:
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()

class TestPose2:
    """
    Command-informed EKF with odometry feedback and a resettable physics twin.

    State x (6):
        [px, py, theta, vx, vy, vtheta]

    Design:
        - Prediction: primary from odometry (true feedback even under collisions).
        - Auxiliary prediction: command-driven physics twin; blended in with small weight.
        - Update: GPS (x,y) + compass (theta) with Mahalanobis gating.
        - After every correction, the physics twin is hard-reset to x to prevent drift.
    """
    def __init__(self, loc_params=LocalizationParams, initial_state: Optional[np.ndarray] = None,
                 twin_blend: float = 0.15, gating_thresh: float = 9.21):
        """
        twin_blend: [0..1] weight for blending the twin-prediction into the odom prediction.
        gating_thresh: chi^2 threshold for innovation gating (≈9.21 ~ 97.5% for 2-3 DoF).
        """
        self.loc = loc_params
        self.nx = 6

        if initial_state is None:
            self.x = np.zeros(self.nx, dtype=float)
        else:
            assert initial_state.shape == (self.nx,)
            self.x = initial_state.astype(float)

        # Physics twin (used as auxiliary predictor; will be reset after updates)
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0.0, 0.0)
        self.space.damping = SPACE_DAMPING
        self.base = DroneBaseTwin()
        self.base._pm_body.position = (float(self.x[0]), float(self.x[1]))
        self.base._pm_body.angle = float(self.x[2])
        self.space.add(self.base._pm_body, *self.base._pm_shapes)

        # Measurement covariance (GPS + compass)
        self.R_full = np.diag([
            self.loc.GPS_NOISE_STD ** 2,
            self.loc.GPS_NOISE_STD ** 2,
            self.loc.COMPASS_NOISE_STD ** 2
        ])

        # Process noise (pose & velocity); velocity noise slightly higher
        self.Q_model = np.diag([0.05, 0.05, 0.01, 0.15, 0.15, 0.08])

        # Odometer noise (for process mapping)
        self.Q_odom = np.diag([
            self.loc.ODOMETER_DISTANCE_NOISE_STD ** 2,
            self.loc.ODOMETER_ALPHA_NOISE_STD ** 2,
            self.loc.ODOMETER_THETA_NOISE_STD ** 2
        ])

        # Covariance
        self.P = np.eye(self.nx) * 1.0

        # Blending & gating params
        self.twin_blend = float(np.clip(twin_blend, 0.0, 1.0))
        self.gating_thresh = float(gating_thresh)

        self.last_command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    @property
    def position(self) -> np.ndarray:
        return self.x[0:2].copy()

    @property
    def orientation(self) -> float:
        return float(self.x[2])

    # ----------------------------- Prediction -----------------------------

    def _predict_with_odometry(self, odometer_values: Tuple[float, float, float]):
        """
        EKF prediction using odometry (primary source of truth; includes wall contacts).
        Odom = (d, alpha_rel, dtheta)
        """
        d, alpha_rel, dtheta = odometer_values
        px, py, theta, vx, vy, vtheta = self.x

        theta_new = normalize_angle(theta + dtheta)
        move_angle = normalize_angle(theta + alpha_rel)

        px_new = px + d * np.cos(move_angle)
        py_new = py + d * np.sin(move_angle)

        # velocity proxies from odom this step (dt=1 sim step)
        vx_new = d * np.cos(move_angle)
        vy_new = d * np.sin(move_angle)
        vtheta_new = dtheta

        x_pred = np.array([px_new, py_new, theta_new, vx_new, vy_new, vtheta_new], dtype=float)

        # Jacobian F wrt state (linearized kinematics like EKF)
        F = np.eye(self.nx)
        F[0, 2] = -d * np.sin(move_angle)   # ∂x/∂theta
        F[1, 2] =  d * np.cos(move_angle)   # ∂y/∂theta
        # simple velocity propagation (keep identity)

        # Map odom noise into pose part via G (6x3)
        G = np.zeros((self.nx, 3))
        # ∂px/∂d, ∂px/∂alpha
        G[0, 0] = np.cos(move_angle)
        G[0, 1] = -d * np.sin(move_angle)
        # ∂py/∂d, ∂py/∂alpha
        G[1, 0] = np.sin(move_angle)
        G[1, 1] =  d * np.cos(move_angle)
        # ∂theta/∂dtheta
        G[2, 2] = 1.0
        # velocities inherit same mapping (coarse but stabilizing)
        G[3, 0] = np.cos(move_angle)
        G[3, 1] = -d * np.sin(move_angle)
        G[4, 0] = np.sin(move_angle)
        G[4, 1] =  d * np.cos(move_angle)
        G[5, 2] = 1.0

        # Propagate covariance
        self.P = F @ self.P @ F.T + G @ self.Q_odom @ G.T + self.Q_model

        self.x = x_pred
        self.x[2] = normalize_angle(self.x[2])

    def _aux_predict_with_twin(self, command: dict):
        """
        Auxiliary prediction using the physics twin; returns predicted state from the twin.
        """
        if command is None:
            command = self.last_command

        # run the twin forward one sim step
        self.base._apply_commands(command)
        for _ in range(SIMULATION_STEPS):
            self.space.step(1.0 / SIMULATION_STEPS)

        px, py = self.base._pm_body.position
        theta = normalize_angle(self.base._pm_body.angle)
        vx, vy = self.base._pm_body.velocity
        vtheta = self.base._pm_body.angular_velocity

        return np.array([px, py, theta, vx, vy, vtheta], dtype=float)

    def _blend_with_twin(self, x_from_odom: np.ndarray, x_from_twin: np.ndarray):
        """
        Blend pose & velocity with a small twin contribution; odom stays dominant.
        """
        if self.twin_blend <= 0.0:
            return x_from_odom
        w = self.twin_blend
        y = (1.0 - w) * x_from_odom + w * x_from_twin
        y[2] = normalize_angle(y[2])
        return y

    # ----------------------------- Update -----------------------------

    def _update_measurements(self,
                             gps_position: Optional[np.ndarray],
                             compass_angle: Optional[float]):
        """
        EKF correction with GPS (x,y) and compass (theta) + Mahalanobis gating.
        """
        z_list = []
        H_rows = []

        if gps_position is not None:
            z_list.extend([float(gps_position[0]), float(gps_position[1])])
            H_rows.append([1, 0, 0, 0, 0, 0])
            H_rows.append([0, 1, 0, 0, 0, 0])

        have_compass = compass_angle is not None
        if have_compass:
            z_list.append(float(compass_angle))
            H_rows.append([0, 0, 1, 0, 0, 0])

        if not z_list:
            return  # no sensors available

        z = np.array(z_list, dtype=float)
        H = np.array(H_rows, dtype=float)
        R = self.R_full[:len(z), :len(z)]

        # Innovation
        hx = H @ self.x
        y = z - hx
        if have_compass:
            y[-1] = normalize_angle(y[-1])

        # Mahalanobis gating
        S = H @ self.P @ H.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # regularize if nearly singular
            S = S + np.eye(S.shape[0]) * 1e-9
            Sinv = np.linalg.inv(S)

        d2 = float(y.T @ Sinv @ y)
        if d2 > self.gating_thresh:
            # reject this update (likely GPS/compass glitch)
            return

        # Kalman update
        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ y
        self.x[2] = normalize_angle(self.x[2])
        I = np.eye(self.nx)
        self.P = (I - K @ H) @ self.P
        # symmetrize
        self.P = 0.5 * (self.P + self.P.T)

    # ----------------------------- Public API -----------------------------

    def step(self,
             odometer_values: Optional[Tuple[float, float, float]] = None,
             gps_position: Optional[np.ndarray] = None,
             compass_angle: Optional[float] = None,
             command: Optional[dict] = None) -> None:
        """
        One filter cycle.
        - Predict with odometry (primary). If none, fall back to twin.
        - Blend a small contribution from the twin.
        - Update with GPS/compass.
        - Reset twin to the corrected state.
        """
        # 1) Primary prediction (odometry)
        used_odom = False
        if odometer_values is not None:
            self._predict_with_odometry(odometer_values)
            used_odom = True

        # 2) Auxiliary twin prediction, blend in
        x_twin = self._aux_predict_with_twin(command if command is not None else self.last_command)
        if used_odom:
            self.x = self._blend_with_twin(self.x, x_twin)
        else:
            # No odom available (e.g., sensor outage) -> rely on twin
            self.x = x_twin

        # 3) Measurement update
        self._update_measurements(gps_position, compass_angle)

        # 4) Reset twin to corrected state to avoid long-term drift
        self.base._pm_body.position = (float(self.x[0]), float(self.x[1]))
        self.base._pm_body.angle = float(self.x[2])
        self.base._pm_body.velocity = (float(self.x[3]), float(self.x[4]))
        self.base._pm_body.angular_velocity = float(self.x[5])

        # 5) Book-keeping
        self.last_command = command if command is not None else self.last_command