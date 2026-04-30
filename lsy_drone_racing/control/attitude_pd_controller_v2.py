"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints 1

import math
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicHermiteSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class AttitudeController_1(Controller):
    """Trajectory following controller using the collective thrust and attitude interface."""

    def create_waypoints(self, obs: dict[str, NDArray[np.floating]]):
        """Generate waypoints for racing through gates.

        Args:
            obs: The current observation containing gate positions and orientations.
        """
        gate_in_offset = 0.23  # metres along gate normal for entry/exit points — TODO: tune
        gate_in_offset_prev = 0.10  # metres along vector from previous waypoint
        gate_out_offset = 0.23  # metres along gate normal for entry/exit points — TODO: tune
        gate_out_offset_next = 0.10  # metres along vector from previous waypoint

        dip_degree = 120

        point_at_obstacle = [True, True, True, False]
        obstacle_ind = np.array([1, 0, 3, 0])
        offset_at_obstacle = np.array(
            [[0.17, -0.17, 0.1], [0.2, -0.2, -0.1], [0.1, 0.2, 0.2], [0, 0, 0]]
        )

        waypoints = [self._start_pos]

        takeoff = self._start_pos.copy()
        takeoff += [0.6, -0.1, 0.3]  # lift to offset as fist waypoint
        waypoints.append(takeoff)

        gates_pos = obs["gates_pos"]  # shape (n_gates, 3)
        gates_quat = obs["gates_quat"]  # shape (n_gates, 4), format [x, y, z, w]

        for i, (pos, quat) in enumerate(zip(gates_pos, gates_quat)):
            normal = R.from_quat(quat).apply([1.0, 0.0, 0.0])
            vec_prev_to_gate = pos - waypoints[-1]
            vec_prev_to_gate_norm = vec_prev_to_gate / np.linalg.norm(vec_prev_to_gate)

            if i + 1 < len(gates_pos):
                vec_gate_to_next_gate = gates_pos[i + 1] - pos
                vec_gate_to_next_gate_norm = vec_gate_to_next_gate / np.linalg.norm(
                    vec_gate_to_next_gate
                )
            else:
                vec_gate_to_next_gate = vec_prev_to_gate  # continue in same direction as bevore
                vec_gate_to_next_gate_norm = vec_prev_to_gate_norm

            # When to perform a dip
            cos_theta = np.dot(normal, vec_gate_to_next_gate_norm)  # bouth are already normed
            theta_dec = np.degrees(np.arccos(cos_theta))

            gate_in_dir_vec = gate_in_offset_prev * vec_prev_to_gate_norm
            gate_out_dir_vec = gate_out_offset_next * vec_gate_to_next_gate_norm

            length_normal_in = np.dot(gate_in_dir_vec, normal)
            length_normal_out = np.dot(gate_out_dir_vec, normal)

            add_normal_in = length_normal_in - gate_in_offset
            add_normal_out = gate_out_offset - length_normal_out

            entry = pos - gate_in_dir_vec + add_normal_in * normal
            if theta_dec < dip_degree:
                exit_ = pos + gate_out_dir_vec + add_normal_out * normal
            else:
                # alternative computation since distance is opposide direction if theta_dec>90
                add_normal_out = gate_out_offset + length_normal_out
                exit_ = pos + gate_out_dir_vec - add_normal_out * normal

            waypoints.append(entry)
            waypoints.append(pos)
            waypoints.append(exit_)

            # if point_at_obstacle[i]:
            #     obs_pos = obs["obstacles_pos"][obstacle_ind[i]].copy()
            #     obs_pos[2] = exit_[2]  # change x pos to previous waypoint

            #     obs_pos = obs_pos + offset_at_obstacle[i]

            #     waypoints.append(obs_pos)
            if point_at_obstacle[i]:
                obs_pos = obs["obstacles_pos"][obstacle_ind[i]].copy()
                obs_pos[2] = exit_[2]
                obs_pos = obs_pos + offset_at_obstacle[i]

                obs_pos = self._push_single_point_away_from_obstacles(obs_pos, obs["obstacles_pos"])

                waypoints.append(obs_pos)

        waypoints = np.array(waypoints)  # shape (1 + n_gates*3, 3)
        if "obstacles_pos" in obs:
            obstacle_pos = np.asarray(obs["obstacles_pos"], dtype=np.float64)
        else:
            obstacle_pos = np.array(
                [[0.0, 0.75, 1.55], [0.95, 0.32, 1.60], [-1.42, -0.18, 1.60], [-0.58, -0.70, 1.60]],
                dtype=np.float64,
            )

        # waypoints = self._push_points_away_from_obstacles(waypoints, obstacle_pos)
        self._waypoints = waypoints


    def _push_single_point_away_from_obstacles(self, point, obstacle_pos):
        pushed = point.copy()
        total_push = np.zeros(2)

        for obs_pos in obstacle_pos:
            offset = pushed[:2] - obs_pos[:2]
            dist = np.linalg.norm(offset)

            if dist < 1e-9:
                continue

            if dist < self.obstacle_safety_radius:
                direction = offset / dist
                strength = self.obstacle_max_push * (1.0 - dist / self.obstacle_safety_radius)
                total_push += strength * direction

        pushed[:2] += total_push
        return pushed

    def create_spline_old(self):
        """Create spline interpolation for waypoints."""
        self._t_total = 10  # s
        t = np.linspace(0, self._t_total, len(self._waypoints))

        cubic_spline = True
        if cubic_spline:
            self._des_pos_spline = CubicSpline(t, self._waypoints)
            self._des_vel_spline = self._des_pos_spline.derivative()
        else:
            # k=3 → cubic B-spline
            bspline = make_interp_spline(t, self._waypoints, k=3)
            self._des_pos_spline = bspline
            self._des_vel_spline = bspline.derivative(1)

    def create_spline(self):
        """Create spline interpolation for waypoints."""
        self._t_total = 6.95  # s

        # Distance-based timing
        distances = np.linalg.norm(np.diff(self._waypoints, axis=0), axis=1)
        cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
        t = self._t_total * cumulative_dist / cumulative_dist[-1]

        

        # cubic_spline = True
        # if cubic_spline:
        #     # Spline with boundary conditions
        #     self._des_pos_spline = CubicSpline(t, self._waypoints)
        #     # alternative if drone should start and end with speed 0
        #     # , bc_type=((1, np.zeros(3)), (1, np.zeros(3))))
        #     self._des_vel_spline = self._des_pos_spline.derivative()
        #     self._des_acc_spline = self._des_vel_spline.derivative()
        # else:
        #     # k=3 → cubic B-spline
        #     bspline = make_interp_spline(t, self._waypoints, k=3)
        #     self._des_pos_spline = bspline
        #     self._des_vel_spline = bspline.derivative(1)
        use_hermite = True

        if use_hermite:
            tangents = np.zeros_like(self._waypoints)

            for i in range(len(self._waypoints)):
                if i == 0:
                    dt = t[1] - t[0]
                    tangents[i] = (self._waypoints[1] - self._waypoints[0]) / max(dt, 1e-6)
                elif i == len(self._waypoints) - 1:
                    dt = t[-1] - t[-2]
                    tangents[i] = (self._waypoints[-1] - self._waypoints[-2]) / max(dt, 1e-6)
                else:
                    dt = t[i + 1] - t[i - 1]
                    tangents[i] = (self._waypoints[i + 1] - self._waypoints[i - 1]) / max(dt, 1e-6)

            tangents *= self.hermite_tangent_scale
            

            self._des_pos_spline = CubicHermiteSpline(t, self._waypoints, tangents)
            self._des_vel_spline = self._des_pos_spline.derivative(1)
            self._des_acc_spline = self._des_pos_spline.derivative(2)

        else:
            self._des_pos_spline = CubicSpline(t, self._waypoints)
            self._des_vel_spline = self._des_pos_spline.derivative()
            self._des_acc_spline = self._des_vel_spline.derivative()

    def _push_points_away_from_obstacles(self, path_points, obstacle_pos):
        pushed = path_points.copy()

        for i in range(len(pushed)):
            total_push = np.zeros(2)

            for obs_pos in obstacle_pos:
                offset = pushed[i, :2] - obs_pos[:2]
                dist = np.linalg.norm(offset)

                if dist < 1e-9:
                    continue

                if dist < self.obstacle_safety_radius:
                    direction = offset / dist
                    strength = self.obstacle_max_push * (1.0 - dist / self.obstacle_safety_radius)
                    total_push += strength * direction

            pushed[i, :2] += total_push

        return pushed

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        
        self._cached_gate_positions = None
        self._cached_obstacle_positions = None
        self._path_dirty = True

        self.hermite_tangent_scale = 0.8

        self.obstacle_safety_radius = 0.23
        self.obstacle_max_push = 0.09
        
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # For more info on the models, check out https://github.com/utiasDSL/drone-models
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]

        PD_var = 2
        if PD_var == 0:
            # original
            self.kp = np.array([0.4, 0.4, 1.25])
            self.ki = np.array([0.05, 0.05, 0.05])
            self.kd = np.array([0.2, 0.2, 0.4])
            self.ki_range = np.array([2.0, 2.0, 0.4])
        elif PD_var == 1:
            # good
            self.kp = np.array([0.8, 0.8, 2.5])
            self.ki = np.array([0.05, 0.05, 0.05])
            self.kd = np.array([0.4, 0.4, 0.8])
            self.ki_range = np.array([2.0, 2.0, 0.4])
        else:
            # without PD
            self.kp = np.array([0.7, 0.7, 2.7])
            self.ki = np.array([0.00, 0.00, 0.00])
            self.kd = np.array([0.4, 0.4, 0.8])
            self.ki_range = np.array([2.0, 2.0, 0.4])

        self.i_error = np.zeros(3)
        self.g = 9.81

        self._start_pos = obs["pos"].copy()  # start at current drone position

        self.create_waypoints(obs)
        self.create_spline()

        self._tick = 0
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:  # Maximum duration reached
            self._finished = True

        # Update splines with current observations
        self.create_waypoints(obs)
        self.create_spline()

        lookahead = 0.12

     

        t_ctrl = min(t + lookahead, self._t_total)
        

        des_pos = self._des_pos_spline(t_ctrl)
        des_vel = self._des_vel_spline(t_ctrl)
        des_acc = self._des_acc_spline(t_ctrl)
        des_yaw = 0.0

        # Calculate the deviations from the desired trajectory
        pos_error = des_pos - obs["pos"]
        vel_error = des_vel - obs["vel"]

        # Update integral error
        self.i_error += pos_error * (1 / self._freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        # Compute target thrust
        target_thrust = np.zeros(3)
        # PID feedback
        target_thrust += self.kp * pos_error
        target_thrust += self.ki * self.i_error
        target_thrust += self.kd * vel_error

        # acceleration feedforward
        acc_norm = np.linalg.norm(des_acc)
        if acc_norm > 8.0:
            des_acc = des_acc / acc_norm * 8.0

        target_thrust += self.drone_mass * des_acc

        # gravity compensation
        target_thrust[2] += self.drone_mass * self.g
        
        # Update z_axis to the current orientation of the drone
        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]

        # update current thrust
        thrust_desired = target_thrust.dot(z_axis)

        # update z_axis_desired
        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        x_c_des = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)

        action = np.concatenate([euler_desired, [thrust_desired]], dtype=np.float32)

        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self.i_error[:] = 0
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        setpoint = self._des_pos_spline(self._tick / self._freq).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        trajectory = self._des_pos_spline(np.linspace(0, self._t_total, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

        draw_points(sim, self._waypoints, rgba=(0.0, 0.0, 1.0, 1.0), size=0.03)
