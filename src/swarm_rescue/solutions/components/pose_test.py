from typing import Optional, Tuple
import numpy as np
import math
from swarm_rescue.simulation.utils.utils import normalize_angle
from solutions.components.pose import PoseEstimator, EKFPoseEstimator
from solutions.utils.dataclasses_config import LocalizationParams
from simulation.utils.definitions import SPACE_DAMPING, PYMUNK_STEPS, ANGULAR_VELOCITY, LINEAR_FORCE
from simulation.utils.constants import ANGULAR_SPEED_RATIO, LINEAR_SPEED_RATIO

class TrueEstimator(PoseEstimator):
    """
    True
    """
    def __init__(self, drone):
        self.drone = drone

    @property
    def position(self) -> np.ndarray:
        return self.drone.true_position().copy()
    
    @property
    def orientation(self) -> float:
        return self.drone.true_angle()
        
    def update(self,
               gps_position: Optional[np.ndarray] = None,
               compass_angle: Optional[float] = None,
               odometer_values: Optional[np.ndarray] = None,
               command: Optional[dict] = None,
               messages: Optional[list] = None) -> None:
        pass

class CommandPoseEstimator(PoseEstimator):
    """
    Command
    """
    def __init__(self, drone):
        self.drone = drone

        self.initialized = False
        self._position = np.zeros(2)
        self._angle = 0.0

        self.angular_ratio = ANGULAR_VELOCITY * ANGULAR_SPEED_RATIO
        self.linear_ratio = LINEAR_FORCE * LINEAR_SPEED_RATIO
        self._mass = 50 * 10

    def _lazy_init(self) -> None:
        """Initialize state from first valid GPS/compass readings."""
        self._position = self.drone.measured_gps_position().copy()
        self._angle = self.drone.measured_compass_angle()
        self._velocity = np.zeros(2)

        self.initialized = True
    
    @property
    def position(self) -> np.ndarray:
        return self._position.copy()
    
    @property
    def orientation(self) -> float:
        return self._angle
        
    def update(self,
               gps_position: Optional[np.ndarray] = None,
               compass_angle: Optional[float] = None,
               odometer_values: Optional[np.ndarray] = None,
               command: Optional[dict] = None,
               messages: Optional[list] = None) -> None:
        if not self.initialized:
            self._lazy_init()
            return
        
        # Angular setup
        angular_velocity = command["rotation"] * self.angular_ratio

        # Linear setup
        # Commands translate to Forces (F = command * ratio)
        cmd_forward = command["forward"]
        cmd_lateral = command["lateral"]
        sqr_norm = cmd_forward ** 2 + cmd_lateral ** 2
        if sqr_norm > 1.0:
            norm = math.sqrt(sqr_norm)
            cmd_forward = cmd_forward / norm
            cmd_lateral = cmd_lateral / norm

        forward_force = cmd_forward * self.linear_ratio
        lateral_force = cmd_lateral * self.linear_ratio

        dt = 1 / PYMUNK_STEPS
        damping_factor = SPACE_DAMPING ** dt

        for _ in range(PYMUNK_STEPS):
            # 1. Rotate local force to global frame using current angle
            # Calculate force first using the current orientation
            c, s = np.cos(self._angle), np.sin(self._angle)
            fx = forward_force * c - lateral_force * s
            fy = forward_force * s + lateral_force * c
            
            # Update Angle
            self._angle += angular_velocity * dt
            angular_velocity *= damping_factor

            # 2. Calculate Acceleration (a = F / m)
            ax = fx / self._mass
            ay = fy / self._mass
            
            # 3. Integrate Velocity (v += a * dt) with Damping
            self._velocity[0] += ax * dt
            self._velocity[0] *= damping_factor
            
            self._velocity[1] += ay * dt
            self._velocity[1] *= damping_factor

            # 4. Integrate Position (p += v * dt)
            self._position[0] += self._velocity[0] * dt
            self._position[1] += self._velocity[1] * dt

        self._angle = normalize_angle(self._angle)

class NewPoseEstimator(PoseEstimator):
    """
    EKF implementation that uses Control Commands (dynamic model) for prediction 
    instead of Odometry alone.
    
    State vector (8):
        x = [px, py, theta, vx, vy, n_gx, n_gy, n_c]^T
    Where:
        - px, py, theta : Pose
        - vx, vy        : Linear velocities in global frame
        - n_gx, n_gy    : AR(1) GPS noise
        - n_c           : AR(1) Compass noise
    """

    def __init__(self, loc_params=LocalizationParams):
        self.loc = loc_params
        self.alpha_ar = float(self.loc.ALPHA_AR1)
        self.nx = 8  # Expanded state dimension

        self.x = np.zeros(self.nx, dtype=float)
        self.initialized = False

        # Physics constants from CommandPoseEstimator
        self.angular_ratio = ANGULAR_VELOCITY * ANGULAR_SPEED_RATIO
        self.linear_ratio = LINEAR_FORCE * LINEAR_SPEED_RATIO
        self._mass = 1.0 # Normalized mass as commonly used in simple physics engines or 0.5kg
        # In CommandPoseEstimator, mass is 50 * 10 = 500? No, let's verify CommandPoseEstimator logic.
        # CommandPoseEstimator: self._mass = 50 * 10 = 500. 
        self._mass = 500.0

        # Measurement Covariance (R) - Stationary noise
        self.R_meas = np.diag([1e-3, 1e-3, 1e-5])
        
        # Process Noise Covariances
        # We assume some noise in the force application and velocity damping
        self.Q_proc = np.diag([0.01, 0.01, 0.01]) # process noise for vx, vy, theta

        # AR(1) innovation variances
        q_g = (1.0 - self.alpha_ar ** 2) * (self.loc.GPS_NOISE_STD ** 2)
        q_c = (1.0 - self.alpha_ar ** 2) * (self.loc.COMPASS_NOISE_STD ** 2)
        self.Q_ar_innov = np.diag([q_g, q_g, q_c])

        self._eps = 1e-9

    def _lazy_init(self, gps_position: np.ndarray, compass_angle: float) -> None:
        self.x = np.zeros(self.nx, dtype=float)
        self.x[0] = gps_position[0]
        self.x[1] = gps_position[1]
        self.x[2] = compass_angle
        # velocities start at 0
        
        P = np.eye(self.nx) * 1e-3
        pose_unc = max(1.0, self.loc.GPS_NOISE_STD)
        P[0, 0] = pose_unc ** 2
        P[1, 1] = pose_unc ** 2
        P[2, 2] = (2.0 * self.loc.COMPASS_NOISE_STD) ** 2
        # AR1 estimates uncertainty
        P[5, 5] = self.loc.GPS_NOISE_STD ** 2
        P[6, 6] = self.loc.GPS_NOISE_STD ** 2
        P[7, 7] = self.loc.COMPASS_NOISE_STD ** 2
        
        self.P = P
        self.initialized = True

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
        
        if not self.initialized:
            if gps_position is not None and compass_angle is not None:
                self._lazy_init(gps_position, compass_angle)
            return

        # 1. Prediction Step using Command (Physics model)
        if command is not None:
            self.predict(command)
        
        # 2. Update Step using Sensors
        # Measurement matrix H mapping state to [x, y, theta]
        # h(x) = [px + n_gx, py + n_gy, theta + n_c]
        # Indexes: px=0, py=1, theta=2, n_gx=5, n_gy=6, n_c=7
        
        H_full = np.zeros((3, self.nx))
        H_full[0, [0, 5]] = 1.0  # x + n_gx
        H_full[1, [1, 6]] = 1.0  # y + n_gy
        H_full[2, [2, 7]] = 1.0  # theta + n_c

        z_list = []
        H_list = []
        R_list = []

        if gps_position is not None:
            z_list.extend([gps_position[0], gps_position[1]])
            H_list.append(H_full[0, :])
            H_list.append(H_full[1, :])
            R_list.extend([self.R_meas[0, 0], self.R_meas[1, 1]])

        if compass_angle is not None:
            z_list.append(compass_angle)
            H_list.append(H_full[2, :])
            R_list.append(self.R_meas[2, 2])

        if len(z_list) > 0:
            z = np.array(z_list)
            H = np.vstack(H_list)
            R = np.diag(R_list)

            hx = H.dot(self.x)
            y = z - hx
            
            # Normalize angle residual if compass is used (last row of H_full relates to theta)
            # Efficient check: if the row corresponds to theta+noise index
            for i in range(len(z_list)):
                if H[i, 2] == 1.0: 
                    y[i] = normalize_angle(y[i])

            S = H.dot(self.P).dot(H.T) + R + np.eye(len(z_list)) * self._eps
            K = self.P.dot(H.T).dot(np.linalg.inv(S))
            
            dx = K.dot(y)
            self.x += dx
            self.x[2] = normalize_angle(self.x[2])

            I = np.eye(self.nx)
            self.P = (I - K.dot(H)).dot(self.P)
            self.P = (self.P + self.P.T) / 2.0

    def predict(self, command: dict) -> None:
        """
        EKF Prediction based on control inputs physics.
        State: [px, py, theta, vx, vy, n_gx, n_gy, n_c]
        """
        # Parse command
        cmd_rot = command.get("rotation", 0.0)
        cmd_fwd = command.get("forward", 0.0)
        cmd_lat = command.get("lateral", 0.0)

        # Normalize linear command
        sqr_norm = cmd_fwd ** 2 + cmd_lat ** 2
        if sqr_norm > 1.0:
            norm = math.sqrt(sqr_norm)
            cmd_fwd /= norm
            cmd_lat /= norm

        # Forces and Constants
        # Note: In Pymunk (simulation), forces are applied in WORLD frame after rotation
        # but the source command is in BODY frame.
        F_body_x = cmd_fwd * self.linear_ratio
        F_body_y = cmd_lat * self.linear_ratio
        
        # Angular velocity is controlled directly (kinematic rotation) in this specific Pymunk model logic
        # provided in CommandPoseEstimator, although typically physics would handle torque.
        # CommandPoseEstimator line: self._angle += angular_velocity * dt
        w_z = cmd_rot * self.angular_ratio

        dt_step = 1.0 / PYMUNK_STEPS
        damping_factor = SPACE_DAMPING ** dt_step
        
        # We will integrate physics for N steps to get next state x_pred
        # However, for Jacobian F, calculating it through a loop is complex.
        # We approximate F by taking the accumulated effect over T = 1 step
        # Since PYMUNK_STEPS is often small (10), we can approximate or aggregate.
        
        # Current State
        px, py, theta, vx, vy = self.x[0:5]
        
        # Physics Integration Loop
        # We must track the state evolution for the 
        # and accumulate Jacobians is possible, or use a simplified transition model for F.
        
        # Simplified Transition Model for Jacobian (Single Step approx with total dt = 1.0)
        # We treat the aggregate of PYMUNK_STEPS as one discrete time update T=1.0
        # But simulation runs physics at higher freq.
        # Let's run the actual physics on x to get x_pred exactly.
        
        curr_px, curr_py, curr_theta = px, py, theta
        curr_vx, curr_vy = vx, vy
        
        # Total damping over 1 sec (since steps occur for 1 sec duration usually in these steps?)
        # Actually simulation usually calls update every tick. 
        # Check definitions: if update is called every logical tick, that is 1.0 unit time?
        # CommandPoseEstimator does: for _ in range(PYMUNK_STEPS): ...
        
        for _ in range(PYMUNK_STEPS):
            c, s = np.cos(curr_theta), np.sin(curr_theta)
            
            # Global forces
            fx = F_body_x * c - F_body_y * s
            fy = F_body_x * s + F_body_y * c
            
            # Update angle
            curr_theta += w_z * dt_step
            # w_z *= damping_factor # In CommandPoseEstimator, ang vel is damped.
            # But command is constant? "angular_velocity = command...". 
            # CommandPoseEstimator applies damping to the variable holding velocity.
            # Here `w_z` comes from command directly every step?
            # CommandPoseEstimator:  inside loop.
            w_z *= damping_factor 

            # Acceleration
            ax = fx / self._mass
            ay = fy / self._mass
            
            # Velocity
            curr_vx += ax * dt_step
            curr_vx *= damping_factor
            
            curr_vy += ay * dt_step
            curr_vy *= damping_factor
            
            # Position
            curr_px += curr_vx * dt_step
            curr_py += curr_vy * dt_step

        curr_theta = normalize_angle(curr_theta)
        
        # Prediction of AR(1) noise states
        n_gx_pred = self.alpha_ar * self.x[5]
        n_gy_pred = self.alpha_ar * self.x[6]
        n_c_pred  = self.alpha_ar * self.x[7]
        
        # Assignments
        self.x[0:5] = [curr_px, curr_py, curr_theta, curr_vx, curr_vy]
        self.x[5:8] = [n_gx_pred, n_gy_pred, n_c_pred]
        
        # ---- Jacobian F Calculation ----
        # Since the loop is non-linear (rotation), exact analytical F matching the loop is messy.
        # We approximate using a standard constant velocity model + some damping.
        # x_k+1 ~= x_k + v_k * T
        # v_k+1 ~= v_k * damp_total
        
        # F matrix (8x8)
        F = np.eye(self.nx)
        
        # Position dependencies
        # p_new = p + v * T (roughly)
        # T is effectively 1.0 here (sum of dt_steps)
        F[0, 3] = 1.0 # dpx/dvx
        F[1, 4] = 1.0 # dpy/dvy
        
        # Velocity dependencies
        # v_new = v * DampingTotal
        total_damping = SPACE_DAMPING # roughly
        F[3, 3] = total_damping
        F[4, 4] = total_damping
        
        # Nonlinear part: The force direction depends on theta.
        # F_global_x = F_body_x cos(theta) - F_body_y sin(theta)
        # dvx/dtheta ~ (1/m) * (-F_bx sin - F_by cos) * T
        c_t, s_t = np.cos(theta), np.sin(theta)
        dFx_dtheta = -F_body_x * s_t - F_body_y * c_t
        dFy_dtheta =  F_body_x * c_t - F_body_y * s_t
        
        F[3, 2] = (dFx_dtheta / self._mass) # dvx/dtheta
        F[4, 2] = (dFy_dtheta / self._mass) # dvy/dtheta
        
        # Also position depends on theta via velocity accumulation
        # dpx/dtheta approx 0.5 * dvx/dtheta * T^2 ? Let's stick to 1st order: 0
        
        # Noise autoregression
        F[5, 5] = self.alpha_ar
        F[6, 6] = self.alpha_ar
        F[7, 7] = self.alpha_ar
        
        # Process Noise Q (8x8)
        # We have noise in velocities (dynamic model error) and noise states
        Q = np.zeros((self.nx, self.nx))
        # Add basic process noise to kinematics/dynamics
        Q[2, 2] = 1e-4 # theta noise
        Q[3, 3] = 1e-3 # vx noise
        Q[4, 4] = 1e-3 # vy noise
        
        # AR innovations
        Q[5:8, 5:8] = self.Q_ar_innov
        
        # Update P
        self.P = F.dot(self.P).dot(F.T) + Q
        self.P = (self.P + self.P.T) / 2.0 + np.eye(self.nx) * self._eps

    def _odometry_update(self, odometer_values):
        # We ignore odometry in this estimator as requested
        pass