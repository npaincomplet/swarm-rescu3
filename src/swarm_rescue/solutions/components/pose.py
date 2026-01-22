import abc
from typing import Tuple, Optional, Dict
import numpy as np
from swarm_rescue.simulation.utils.utils import normalize_angle
from solutions.utils.dataclasses_config import *

class PoseEstimator(abc.ABC):
    """
    Abstract base class for pose estimation.
    Subclasses must implement the update method and position/orientation properties.
    """

    @property
    @abc.abstractmethod
    def position(self) -> np.ndarray:
        """Get the current position as a numpy array."""
        pass
    
    @property
    @abc.abstractmethod
    def orientation(self) -> float:
        """Get the current orientation as a float."""
        pass
    
    @abc.abstractmethod
    def update(self,
               gps_position: Optional[np.ndarray] = None,
               compass_angle: Optional[float] = None,
               odometer_values: Optional[np.ndarray] = None,
               command: Optional[dict] = None,
               messages: Optional[list] = None) -> None:
        """
        Update the pose based on available sensor data and commands.
        Subclasses should implement this method to handle their specific logic.
        """
        pass

class SimplePoseEstimator(PoseEstimator):
    """
    Simple pose estimation using GPS and dead reckoning.
    """
    def __init__(self, size_area: Tuple[float, float]):
        self._position = np.zeros(2)
        self._orientation = 0.0
        self.size_area = size_area
    
    @property
    def position(self) -> np.ndarray:
        return self._position
    
    @property
    def orientation(self) -> float:
        return self._orientation
        
    def update(self,
               gps_position: Optional[np.ndarray] = None,
               compass_angle: Optional[float] = None,
               odometer_values: Optional[np.ndarray] = None,
               command: Optional[dict] = None,
               messages: Optional[list] = None) -> None:

        if gps_position is not None and compass_angle is not None:
            self._gps_compass_update(gps_position, compass_angle)
        
        elif odometer_values:
            self._odometry_update(odometer_values)

    
    def _gps_compass_update(self,
               gps_position: Optional[np.ndarray] = None,
               compass_angle: Optional[float] = None) -> None:
        
        self._position = gps_position.copy()
        self._orientation = compass_angle

    def _odometry_update(self,
               odometer_values: Optional[Tuple[float, float, float]] = None) -> None:
        
        self._orientation += odometer_values[2]
        self._orientation = normalize_angle(self._orientation)
        
        move_angle = normalize_angle(odometer_values[1] + self._orientation)
        self._position[0] += np.cos(move_angle) * odometer_values[0]
        self._position[1] += np.sin(move_angle) * odometer_values[0]

class EKFPoseEstimator(PoseEstimator):
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

    def __init__(self, loc_params=LocalizationParams, initial_state: Optional[np.ndarray] = None):
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

    def update(self,
               gps_position: Optional[np.ndarray] = None,
               compass_angle: Optional[float] = None,
               odometer_values: Optional[np.ndarray] = None,
               command: Optional[dict] = None,
               messages: Optional[list] = None) -> None:
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

    def predict(self, odometer_values: np.ndarray) -> None:
        """
        EKF prediction step using odometry and AR(1) evolution of noise states.

        odometer_values: np.array([d, alpha_rel, dtheta])
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

class PoseEstimatorManager:
    estimators: Dict[str, PoseEstimator]
    active_name: str

    def __init__(self):
        self.estimators = {}
        self.active_name = None

    def add(self, name: str, estimator: PoseEstimator, active: bool = False) -> None:
        self.estimators[name] = estimator
        if active or not self.active_name:
            self.active_name = name

    def set_active(self, name: str) -> None:
        if name in self.estimators:
            self.active_name = name

    @property
    def active(self) -> Optional[PoseEstimator]:
        return self.estimators.get(self.active_name)

    def update_all(self,
                   gps_position: Optional[np.ndarray] = None,
                   compass_angle: Optional[float] = None,
                   odometer_values: Optional[np.ndarray] = None,
                   command: Optional[dict] = None,
                   messages: Optional[list] = None) -> None:
        for estimator in self.estimators.values():
            estimator.update(
                gps_position=gps_position,
                compass_angle=compass_angle,
                odometer_values=odometer_values,
                command=command,
                messages=messages
            )