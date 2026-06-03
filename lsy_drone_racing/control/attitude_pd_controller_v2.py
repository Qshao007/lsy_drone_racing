"""Structured attitude controller for LSY drone racing.

This controller keeps path planning and control separated:

1. RawObservation parses obs/info.
2. SelfLocation maintains pose and coordinate transforms.
3. LocalMapping maintains an estimated gate/obstacle map.
4. TrajectoryManager produces an entry-center-exit reference.
5. This controller only tracks the reference with a PD attitude/thrust law.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.Path_planner.local_mapping import LocalMapping
from lsy_drone_racing.Path_planner.raw_observation import RawObservation
from lsy_drone_racing.Path_planner.self_location import SelfLocation
from lsy_drone_racing.Path_planner.trajectory_manager import TrajectoryManager

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class AttitudeController_1(Controller):
    """Reference-tracking controller using the collective thrust + attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._freq = config.env.freq
        self._tick = 0
        self._finished = False
        self.debug = os.getenv("LSY_DEBUG_PATH", "0") == "1"
        self._actual_path_history: list[NDArray[np.floating]] = [obs["pos"].copy()]

        # ------------------------------------------------------------------
        # Path-planning pipeline
        # ------------------------------------------------------------------
        self.config_path = self._infer_config_path(config)

        self.raw_observation = RawObservation(self.config_path)
        self.self_location = SelfLocation()
        self.local_mapping = LocalMapping(self.raw_observation.config)
        is_random_track = bool(config.env.track.get("randomize", False))
        self.trajectory_manager = TrajectoryManager(
            waypoint_threshold=0.20,
            entry_distance=0.42,
            exit_distance=0.58,
            max_speed=0.25,
            max_segment_length=0.45,
            obstacle_safety_radius=0.36 if is_random_track else 0.32,
            obstacle_detour_margin=0.08,
            enable_dynamic_replan=is_random_track,
            loop=True,
            debug=self.debug,
        )

        self._last_map_signature: tuple | None = None
        self._last_ref = None
        self._last_waypoints = np.empty((0, 3), dtype=float)
        self._last_target_gate = 0
        self._last_velocity = np.asarray(obs.get("vel", np.zeros(3)), dtype=float)
        self._estimated_acc_world = np.zeros(3, dtype=float)

        # ------------------------------------------------------------------
        # Drone parameters
        # ------------------------------------------------------------------
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = float(drone_params["mass"])
        self.thrust_min = float(drone_params["thrust_min"] * 4.0)
        self.thrust_max = float(drone_params["thrust_max"] * 4.0)

        # Conservative attitude-mode PID gains. Keep I small: it trims slow
        # bias without dragging the drone sideways after a gate switch.
        self.kp = np.array([0.45, 0.45, 1.10])
        self.ki = np.array([0.018, 0.018, 0.030])
        self.kd = np.array([0.28, 0.28, 0.48])
        self.ki_range = np.array([0.45, 0.45, 0.20])
        self.i_error = np.zeros(3)

        self.g = 9.81
        self.tilt_limit_rad = np.deg2rad(35.0)
        self.lateral_acc_limit = 5.0
        self.vertical_acc_limit = 5.0
        self.acc_limit = 4.0
        self.position_error_limit = 0.55

        self._previous_action = np.array(
            [0.0, 0.0, 0.0, self.drone_mass * self.g],
            dtype=np.float32,
        )

    # ======================================================================
    # Main control loop
    # ======================================================================

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None,
    ) -> NDArray[np.floating]:
        """Compute [roll_des, pitch_des, yaw_des, thrust_des]."""
        if self._race_finished(obs):
            self._finished = True
            return self._previous_action

        planner_obs = self._build_planner_observation(obs)

        parsed_obs = self.raw_observation.update(planner_obs, info)
        self_state = self.self_location.update(parsed_obs)
        self_state["acc_world"] = self._estimate_acceleration(obs)
        estimated_map = self.local_mapping.update(parsed_obs, self.self_location)

        map_signature = self._map_signature(estimated_map)

        if map_signature != self._last_map_signature:
            if self.trajectory_manager.initialized:
                self.trajectory_manager.update_map_if_needed(estimated_map)
            self._last_map_signature = map_signature

        self._sync_trajectory_with_environment(obs)
        current_target_gate = self._current_target_gate(obs)

        ref = self.trajectory_manager.update(
            self_state,
            estimated_map,
            target_gate=current_target_gate,
        )

        self._last_ref = ref
        self._last_waypoints = (
            np.asarray(
                [wp["pos"] for wp in self.trajectory_manager.waypoints],
                dtype=float,
            )
            if self.trajectory_manager.waypoints
            else np.empty((0, 3), dtype=float)
        )

        des_pos = np.asarray(ref["pos"], dtype=float)
        des_vel = np.asarray(ref["vel"], dtype=float)
        des_acc = np.asarray(ref["acc"], dtype=float)
        ref_yaw = float(ref["yaw"])

        pos = np.asarray(obs["pos"], dtype=float)
        vel = np.asarray(obs["vel"], dtype=float)
        quat = np.asarray(obs["quat"], dtype=float)

        current_rpy = R.from_quat(quat).as_euler("xyz", degrees=False)
        current_roll = float(current_rpy[0])
        current_pitch = float(current_rpy[1])
        current_yaw = float(current_rpy[2])

        # Important:
        # For now, keep yaw close to current yaw to avoid sudden 180-degree yaw flips.
        # Gate yaw can jump to around -pi after passing Gate0, which destabilizes attitude control.
        des_yaw = current_yaw

        pos_error_raw = des_pos - pos
        pos_error_norm = np.linalg.norm(pos_error_raw)
        if pos_error_norm > self.position_error_limit:
            pos_error = pos_error_raw / pos_error_norm * self.position_error_limit
        else:
            pos_error = pos_error_raw
        vel_error = des_vel - vel

        self.i_error += pos_error * (1.0 / self._freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        acc_norm = np.linalg.norm(des_acc)
        if acc_norm > self.acc_limit:
            des_acc = des_acc / acc_norm * self.acc_limit

        target_force = np.zeros(3)
        target_force += self.kp * pos_error
        target_force += self.ki * self.i_error
        target_force += self.kd * vel_error
        target_force += self.drone_mass * des_acc
        target_force[2] += self.drone_mass * self.g

        target_force = self._limit_target_force(target_force)

        euler_desired, thrust_desired = self._force_to_attitude_action(
            target_force,
            quat,
            des_yaw,
        )

        action = np.concatenate(
            [euler_desired, [thrust_desired]],
            dtype=np.float32,
        )

        self._previous_action = action

        if self.debug and self._tick % 10 == 0:
            print(
                "\n========== DRONE STATE DEBUG =========="
                "\ntarget_gate =", current_target_gate,
                "\nwp_id       =", ref["wp_id"],
                "\nwp_type     =", ref["wp_type"],
                "\nwp_gate     =", ref["gate_id"],
                "\npos         =", np.round(pos, 3),
                "\nvel         =", np.round(vel, 3),
                "\nrpy         =", np.round(current_rpy, 3),
                "\ncurrent_yaw =", round(current_yaw, 3),
                "\nref_yaw     =", round(ref_yaw, 3),
                "\nused_yaw    =", round(des_yaw, 3),
                "\ndes_pos     =", np.round(des_pos, 3),
                "\ndes_vel     =", np.round(des_vel, 3),
                "\npos_error   =", np.round(pos_error, 3),
                "\npos_error_raw =", np.round(pos_error_raw, 3),
                "\nvel_error   =", np.round(vel_error, 3),
                "\nforce       =", np.round(target_force, 3),
                "\neuler_des   =", np.round(euler_desired, 3),
                "\nthrust_des  =", round(float(thrust_desired), 3),
                "\naction      =", np.round(action, 3),
                "\n=======================================\n"
            )

        return action
        
        
    def _track_reference(
        self,
        obs: dict[str, NDArray[np.floating]],
        des_pos: NDArray[np.floating],
        des_vel: NDArray[np.floating],
        des_acc: NDArray[np.floating],
        des_yaw: float,
    ) -> NDArray[np.floating]:
        """PD position tracking with acceleration feedforward."""
        pos = np.asarray(obs["pos"], dtype=float)
        vel = np.asarray(obs.get("vel", np.zeros(3)), dtype=float)

        pos_error_raw = des_pos - pos
        pos_error_norm = np.linalg.norm(pos_error_raw)
        if pos_error_norm > self.position_error_limit:
            pos_error = pos_error_raw / pos_error_norm * self.position_error_limit
        else:
            pos_error = pos_error_raw
        vel_error = des_vel - vel

        self.i_error += pos_error * (1.0 / self._freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        acc_norm = np.linalg.norm(des_acc)
        if acc_norm > self.acc_limit:
            des_acc = des_acc / acc_norm * self.acc_limit

        target_force = np.zeros(3)
        target_force += self.kp * pos_error
        target_force += self.ki * self.i_error
        target_force += self.kd * vel_error
        target_force += self.drone_mass * des_acc
        target_force[2] += self.drone_mass * self.g

        target_force = self._limit_target_force(target_force)
        euler_desired, thrust_desired = self._force_to_attitude_action(
            target_force,
            obs["quat"],
            des_yaw,
        )

        return np.concatenate([euler_desired, [thrust_desired]], dtype=np.float32)

    # ======================================================================
    # Observation conversion for planner modules
    # ======================================================================

    def _estimate_acceleration(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        vel = np.asarray(obs.get("vel", np.zeros(3)), dtype=float)
        raw_acc = (vel - self._last_velocity) * float(self._freq)
        self._last_velocity = vel.copy()

        acc_norm = np.linalg.norm(raw_acc)
        if acc_norm > self.acc_limit:
            raw_acc = raw_acc / acc_norm * self.acc_limit

        alpha = 0.35
        self._estimated_acc_world = (
            (1.0 - alpha) * self._estimated_acc_world
            + alpha * raw_acc
        )
        return self._estimated_acc_world.copy()

    @staticmethod
    def _infer_config_path(config: dict) -> str:
        """Recover the TOML path needed by the path-planner helper classes."""
        track = config.env.track

        if bool(track.get("randomize", False)):
            return "config/level3.toml"

        return "config/level2.toml"

    def _build_planner_observation(
        self,
        obs: dict[str, NDArray[np.floating]],
    ) -> dict:
        """Convert environment obs into the unified format expected by RawObservation.

        The environment usually provides gates_pos + gates_quat. RawObservation
        works best when each gate already contains pos/rpy/yaw, so we create
        that normalized representation here.
        """
        planner_obs = dict(obs)

        gates_pos = np.asarray(obs.get("gates_pos", []), dtype=float)
        gates_quat = np.asarray(obs.get("gates_quat", []), dtype=float)

        gates = []
        for gate_id, pos in enumerate(gates_pos):
            if gate_id < len(gates_quat):
                rpy = R.from_quat(gates_quat[gate_id]).as_euler("xyz", degrees=False)
            else:
                rpy = np.zeros(3)

            gates.append(
                {
                    "pos": np.asarray(pos, dtype=float),
                    "rpy": np.asarray(rpy, dtype=float),
                    "yaw": float(rpy[2]),
                }
            )

        if gates:
            planner_obs["gates"] = gates

        obstacles_pos = np.asarray(obs.get("obstacles_pos", []), dtype=float)
        obstacles = [{"pos": np.asarray(pos, dtype=float)} for pos in obstacles_pos]
        if obstacles:
            planner_obs["obstacles"] = obstacles

        return planner_obs

    def _map_signature(self, estimated_map: dict) -> tuple:
        """Compact signature to detect map changes."""
        gate_sig = tuple(
            (
                gate["id"],
                tuple(np.round(gate["pos_world"], 3)),
                round(float(gate["yaw"]), 3),
                gate["source"],
            )
            for gate in estimated_map["gates"]
        )
        obstacle_sig = tuple(
            (
                obstacle["id"],
                tuple(np.round(obstacle["pos_world"], 3)),
                obstacle["source"],
            )
            for obstacle in estimated_map["obstacles"]
        )
        return gate_sig, obstacle_sig

    def _race_finished(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        target_gate = int(obs.get("target_gate", 0))
        n_gates = len(obs.get("gates_pos", []))
        return target_gate < 0 or (n_gates > 0 and target_gate >= n_gates)

    def _current_target_gate(self, obs: dict[str, NDArray[np.floating]]) -> int:
        """Return the next gate index reported by the environment."""
        target_gate = int(obs.get("target_gate", self._last_target_gate))
        if target_gate < 0:
            return len(obs.get("gates_pos", []))
        return target_gate

    def _sync_trajectory_with_environment(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Synchronize waypoint progress with the simulator's target_gate.

        The waypoint manager uses distance thresholds. At speed, the drone can
        pass through a gate without getting close enough to the exact center
        waypoint. The simulator may already switch target_gate to the next gate,
        while the waypoint manager still wants to go back to the previous gate.
        This function prevents that by trusting target_gate as the authoritative
        gate-progress signal.
        """
        current_target_gate = self._current_target_gate(obs)

        if current_target_gate > self._last_target_gate:
            self.i_error[:] = 0.0
            self._set_waypoint_to_gate(current_target_gate)

        self._last_target_gate = current_target_gate

    def _set_waypoint_to_gate(self, gate_id: int) -> None:
        """Move TrajectoryManager to the entry waypoint of gate_id."""
        if not self.trajectory_manager.waypoints:
            return

        # TrajectoryManager creates three waypoints for each gate:
        # entry, center, exit.
        wp_id = 3 * int(gate_id)
        wp_id = int(np.clip(wp_id, 0, len(self.trajectory_manager.waypoints) - 1))
        self.trajectory_manager.current_wp_id = wp_id

    # ======================================================================
    # Attitude/thrust utilities
    # ======================================================================

    def _limit_target_force(
        self,
        force: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Keep the force command inside attitude-controller feasible limits."""
        limited = np.asarray(force, dtype=float).copy()

        min_vertical_force = max(self.thrust_min, 0.20 * self.drone_mass * self.g)
        limited[2] = np.clip(limited[2], min_vertical_force, self.thrust_max)

        lateral_norm = np.linalg.norm(limited[:2])
        lateral_force_limit = min(
            self.drone_mass * self.lateral_acc_limit,
            math.tan(self.tilt_limit_rad) * max(limited[2], 1e-6),
        )
        if lateral_norm > lateral_force_limit:
            limited[:2] *= lateral_force_limit / (lateral_norm + 1e-9)

        return limited

    def _force_to_attitude_action(
        self,
        force: NDArray[np.floating],
        quat: NDArray[np.floating],
        yaw: float,
    ) -> tuple[NDArray[np.floating], float]:
        """Convert a desired world-frame force to roll, pitch, yaw, thrust."""
        force_norm = np.linalg.norm(force)
        if force_norm < 1e-6:
            force = np.array([0.0, 0.0, self.drone_mass * self.g], dtype=float)
            force_norm = np.linalg.norm(force)

        z_axis_desired = force / force_norm
        z_axis_desired = self._limit_body_z_tilt(z_axis_desired)

        x_c_des = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_norm = np.linalg.norm(y_axis_desired)
        if y_axis_norm < 1e-6:
            y_axis_desired = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            y_axis_desired /= y_axis_norm
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        r_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(r_desired).as_euler("xyz", degrees=False)
        euler_desired[0] = np.clip(euler_desired[0], -self.tilt_limit_rad, self.tilt_limit_rad)
        euler_desired[1] = np.clip(euler_desired[1], -self.tilt_limit_rad, self.tilt_limit_rad)
        euler_desired[2] = yaw

        current_z_axis = R.from_quat(quat).as_matrix()[:, 2]
        thrust_desired = float(np.dot(force, current_z_axis))
        thrust_desired = float(np.clip(thrust_desired, self.thrust_min, self.thrust_max))
        return euler_desired, thrust_desired

    def _limit_body_z_tilt(
        self,
        desired_z_axis: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Limit commanded body-z tilt away from world z."""
        world_z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        desired_z_axis = desired_z_axis / (np.linalg.norm(desired_z_axis) + 1e-9)
        tilt_angle = np.arccos(np.clip(np.dot(desired_z_axis, world_z_axis), -1.0, 1.0))

        if tilt_angle <= self.tilt_limit_rad:
            return desired_z_axis

        horizontal_part = desired_z_axis.copy()
        horizontal_part[2] = 0.0
        horizontal_norm = np.linalg.norm(horizontal_part)
        if horizontal_norm < 1e-9:
            return world_z_axis

        horizontal_part = horizontal_part / horizontal_norm * np.sin(self.tilt_limit_rad)
        limited_z_axis = np.array(
            [horizontal_part[0], horizontal_part[1], np.cos(self.tilt_limit_rad)],
            dtype=float,
        )
        return limited_z_axis / np.linalg.norm(limited_z_axis)

    # ======================================================================
    # Callbacks
    # ======================================================================

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1
        self._actual_path_history.append(obs["pos"].copy())
        if len(self._actual_path_history) > 600:
            self._actual_path_history = self._actual_path_history[-600:]

        if self.debug and self._tick <= 100 and self._tick % 10 == 0:
            recent_path = np.asarray(self._actual_path_history[-5:], dtype=float)
            print(
                "[ACTUAL_PATH]",
                f"step={self._tick}",
                "last5=",
                np.round(recent_path, 3).tolist(),
            )

        return self._finished

    def episode_callback(self):
        if self.debug and self._actual_path_history:
            try:
                with open("actual_path_history.txt", "w") as f:
                    for pos in self._actual_path_history:
                        f.write(
                            ",".join(map(str, np.round(pos, 4).tolist())) + "\n"
                        )
                print(
                    f"[ACTUAL_PATH] saved {len(self._actual_path_history)} points to actual_path_history.txt"
                )
            except OSError as exc:
                print("[ACTUAL_PATH] failed to save actual_path_history.txt:", exc)

        self.i_error[:] = 0.0
        self._tick = 0
        self._finished = False
        self._actual_path_history = []
        self.trajectory_manager.initialized = False
        self.trajectory_manager.waypoints = []
        self.trajectory_manager.current_wp_id = 0
        self._last_map_signature = None
        self._last_ref = None
        self._last_waypoints = np.empty((0, 3), dtype=float)
        self._last_target_gate = 0

    def render_callback(self, sim: Sim):
        if self._last_ref is not None:
            draw_points(
                sim,
                np.asarray(self._last_ref["pos"], dtype=float).reshape(1, -1),
                rgba=(1.0, 0.0, 0.0, 1.0),
                size=0.03,
            )

        if len(self._last_waypoints) >= 2:
            draw_line(sim, self._last_waypoints, rgba=(0.0, 1.0, 0.0, 1.0))
            draw_points(sim, self._last_waypoints, rgba=(0.0, 0.0, 1.0, 1.0), size=0.025)

        if len(self._actual_path_history) >= 2:
            actual_path = np.asarray(self._actual_path_history, dtype=float)
            draw_line(sim, actual_path, rgba=(1.0, 0.85, 0.0, 1.0))
