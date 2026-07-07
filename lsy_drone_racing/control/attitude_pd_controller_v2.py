"""Structured attitude controller for LSY drone racing.

This controller keeps path planning and control separated:

1. RawObservation parses obs/info.
2. SelfLocation maintains pose and coordinate transforms.
3. LocalMapping maintains an estimated gate/obstacle map.
4. TrajectoryManager produces safe gate waypoints and velocity references.
5. This controller tracks the reference with a PD attitude/thrust law.

Two operating modes are used:
- fixed_fast_mode: fixed seed-2026 map, tuned for speed and repeatability.
- random_online_mode: randomized map, tuned more conservatively for robustness.
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
        """Initialize mapping, planning, and attitude-control state."""
        super().__init__(obs, info, config)

        self._freq = config.env.freq
        self._tick = 0
        self._finished = False
        self.debug = os.getenv("LSY_DEBUG_PATH", "0") == "1"
        self._actual_path_history: list[NDArray[np.floating]] = [obs["pos"].copy()]

        # ------------------------------------------------------------------
        # Mode selection and path-planning pipeline
        # ------------------------------------------------------------------
        self.config_path = self._infer_config_path(config)
        self._use_fixed_seed_map = os.path.basename(self.config_path) == "level3_1.toml"

        self.raw_observation = RawObservation(self.config_path)
        self.self_location = SelfLocation()
        self.local_mapping = LocalMapping(self.raw_observation.config)
        is_random_track = bool(config.env.track.get("randomize", False))
        self._is_random_track = is_random_track
        random_online_mode = is_random_track and not self._use_fixed_seed_map
        fixed_fast_mode = self._use_fixed_seed_map
        self._fixed_fast_mode = fixed_fast_mode

        # TrajectoryManager owns waypoint generation, obstacle/gate detours,
        # speed scheduling, and fixed-map fast-mode shortcuts.
        self.trajectory_manager = TrajectoryManager(
            waypoint_threshold=0.20,
            entry_distance=0.50 if random_online_mode else 0.42,
            exit_distance=0.62 if random_online_mode else 0.58,
            max_speed=0.24 if random_online_mode else (1.50 if fixed_fast_mode else 0.25),
            max_segment_length=0.45,
            obstacle_safety_radius=0.36 if is_random_track else 0.32,
            obstacle_detour_margin=0.08,
            gate_align_distance=0.24 if random_online_mode else 0.18,
            gate_pass_speed_scale=(
                0.52 if random_online_mode else (0.76 if fixed_fast_mode else 0.58)
            ),
            enable_dynamic_replan=is_random_track,
            enable_previous_gate_recross=True,
            previous_gate_recross_until_gate=3 if self._use_fixed_seed_map else None,
            aggressive_speed=fixed_fast_mode,
            loop=True,
            debug=self.debug,
        )

        # Runtime planner/control state. These caches avoid unnecessary replans
        # and smooth short waypoint/reference jumps.
        self._last_map_signature: tuple | None = None
        self._last_ref = None
        self._last_waypoints = np.empty((0, 3), dtype=float)
        self._last_target_gate = 0
        self._last_velocity = np.asarray(obs.get("vel", np.zeros(3)), dtype=float)
        self._estimated_acc_world = np.zeros(3, dtype=float)
        self._smoothed_ref_pos: NDArray[np.floating] | None = None
        self._smoothed_ref_vel: NDArray[np.floating] | None = None
        self._last_ref_key: tuple | None = None
        # Search/staging state used only when the current gate is not yet seen
        # in randomized online mode.
        self._exploration_wp_id: int | None = None
        self._search_target_gate: int | None = None
        self._search_wp_start_tick = 0
        self._staged_target_gate: int | None = None
        self._staging_start_tick = 0
        self.search_min_altitude = 0.62
        self.search_max_altitude = 1.30
        self.search_gate_clearance = 0.08
        self.first_gate_search_altitude = 0.68
        self.search_step_xy = 0.82
        self.search_speed_xy = 0.42
        self.followup_search_step_xy = 0.52
        self.followup_search_speed_xy = 0.26
        self.search_climb_speed = 0.20
        self.search_waypoint_threshold = 0.42
        self.search_waypoint_timeout_s = 1.35
        self.search_horizontal_start_alt = 0.58
        self.search_horizontal_ramp = 0.18
        self._search_xy_waypoints = np.asarray(
            [
                [0.00, 0.00],
                [1.80, 0.00],
                [1.80, 0.75],
                [0.00, 0.75],
                [-1.80, 0.75],
                [-1.80, 0.00],
                [-1.80, -0.75],
                [0.00, -0.75],
                [1.80, -0.75],
            ],
            dtype=float,
        )

        # ------------------------------------------------------------------
        # Drone parameters
        # ------------------------------------------------------------------
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = float(drone_params["mass"])
        self.thrust_min = float(drone_params["thrust_min"] * 4.0)
        self.thrust_max = float(drone_params["thrust_max"] * 4.0)

        # Attitude-mode PID gains. Fixed-map mode is allowed to be faster;
        # random online mode remains conservative to absorb perception changes.
        self.kp = np.array([0.66, 0.66, 1.22]) if fixed_fast_mode else np.array([0.45, 0.45, 1.10])
        self.ki = np.array([0.018, 0.018, 0.030])
        self.kd = np.array([0.40, 0.40, 0.56]) if fixed_fast_mode else np.array([0.28, 0.28, 0.48])
        self.ki_range = np.array([0.45, 0.45, 0.20])
        self.i_error = np.zeros(3)

        self.g = 9.81
        self.tilt_limit_rad = np.deg2rad(42.0 if fixed_fast_mode else 35.0)
        self.lateral_acc_limit = 6.2 if fixed_fast_mode else 5.0
        self.vertical_acc_limit = 5.0
        self.acc_limit = 4.0
        self.position_error_limit = 0.68 if fixed_fast_mode else 0.55

        # Reference/action smoothing reduces visible oscillation on dense
        # detour paths while keeping gate precision waypoints responsive.
        self.ref_pos_smoothing_alpha = 0.62 if fixed_fast_mode else 0.52
        self.ref_vel_smoothing_alpha = 0.50 if fixed_fast_mode else 0.42
        self.max_attitude_step = np.deg2rad(9.0 if fixed_fast_mode else 6.0)
        self.max_yaw_step = np.deg2rad(10.0)
        self.max_thrust_step = (0.24 if fixed_fast_mode else 0.18) * self.drone_mass * self.g

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

        # 1) Normalize simulator observations for the mapping/planning modules.
        planner_obs = self._build_planner_observation(obs)
        parsed_obs = self.raw_observation.update(planner_obs, info)
        self_state = self.self_location.update(parsed_obs)
        self_state["acc_world"] = self._estimate_acceleration(obs)
        estimated_map = self.local_mapping.update(parsed_obs, self.self_location)

        # 2) Choose the map used for planning. Fixed seed uses the full known
        # map; random online mode plans only with currently observed objects.
        current_target_gate = self._current_target_gate(obs)
        planning_map = self._planning_map(estimated_map, current_target_gate)
        map_signature = self._map_signature(planning_map)

        if map_signature != self._last_map_signature:
            if self.trajectory_manager.initialized and planning_map["gates"]:
                self.trajectory_manager.update_map_if_needed(planning_map)
            self._last_map_signature = map_signature

        self._sync_trajectory_with_environment(obs)

        # 3) Produce a reference. If no gate is visible in random mode, search;
        # otherwise stage in front of the gate before tracking the full route.
        if planning_map["gates"]:
            staging_ref = self._gate_staging_reference(
                self_state,
                planning_map,
                current_target_gate,
            )
            if staging_ref is not None:
                ref = staging_ref
            else:
                ref = self.trajectory_manager.update(
                    self_state,
                    planning_map,
                    target_gate=current_target_gate,
                )
        else:
            ref = self._exploration_reference(self_state, estimated_map, current_target_gate)

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

        # 4) Track the reference with a force-level PD law, then convert the
        # desired world-frame force into attitude + collective thrust.
        pos = np.asarray(obs["pos"], dtype=float)
        vel = np.asarray(obs["vel"], dtype=float)
        quat = np.asarray(obs["quat"], dtype=float)

        current_rpy = R.from_quat(quat).as_euler("xyz", degrees=False)
        current_yaw = float(current_rpy[2])

        # Important:
        # For now, keep yaw close to current yaw to avoid sudden 180-degree yaw flips.
        # Gate yaw can jump to around -pi after passing Gate0, which destabilizes attitude control.
        des_yaw = current_yaw

        des_pos, des_vel = self._smooth_reference(ref, des_pos, des_vel, pos)

        pos_error_raw = des_pos - pos
        pos_error_norm = np.linalg.norm(pos_error_raw)
        if pos_error_norm > self.position_error_limit:
            pos_error = pos_error_raw / pos_error_norm * self.position_error_limit
        else:
            pos_error = pos_error_raw
        vel_error = des_vel - vel

        if str(ref.get("wp_type", "")).endswith("_route_search"):
            self.i_error *= 0.95
        else:
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
        action = self._limit_action_rate(action, ref)

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
        """Legacy standalone PD tracker kept for quick controller experiments."""
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
        """Estimate world acceleration from measured velocity with light filtering."""
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
            # For the final fixed-map test, keep the planner helpers consistent
            # with the simulator config. Fall back to level3.toml for the
            # normal randomized level-3 setting.
            seed = str(config.env.get("seed", ""))
            if seed == "2026" and os.path.exists("config/level3_1.toml"):
                return "config/level3_1.toml"
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

    def _planning_map(self, estimated_map: dict, target_gate: int) -> dict:
        """Select fixed-map or currently visible online-map data for planning."""
        if not self._is_random_track or self._use_fixed_seed_map:
            return estimated_map

        gates = [
            gate
            for gate in estimated_map.get("gates", [])
            if gate.get("seen", False) and int(gate.get("id", -1)) == int(target_gate)
        ]
        obstacles = [
            obstacle
            for obstacle in estimated_map.get("obstacles", [])
            if obstacle.get("seen", False)
        ]
        return {"gates": gates, "obstacles": obstacles}

    def _exploration_reference(
        self,
        self_state: dict,
        estimated_map: dict,
        target_gate: int,
    ) -> dict:
        """Generate a slow search reference until the current random gate is seen."""
        pos = np.asarray(self_state["pos_world"], dtype=float)
        target_altitude = self._search_altitude_for_gate(target_gate)

        if self._search_target_gate != target_gate:
            self._search_target_gate = target_gate
            self._exploration_wp_id = None
            self._staged_target_gate = None
            self._staging_start_tick = self._tick

        if self._exploration_wp_id is None:
            distances = np.linalg.norm(self._search_xy_waypoints - pos[:2], axis=1)
            self._exploration_wp_id = int(np.argmin(distances))
            self._search_wp_start_tick = self._tick

        self._exploration_wp_id %= len(self._search_xy_waypoints)
        target_xy = self._search_xy_waypoints[self._exploration_wp_id].copy()
        reached_wp = np.linalg.norm(target_xy - pos[:2]) < self.search_waypoint_threshold
        timed_out = (
            self._tick - self._search_wp_start_tick
            > int(self.search_waypoint_timeout_s * self._freq)
            and pos[2] > self.search_horizontal_start_alt
        )
        if reached_wp or timed_out:
            self._exploration_wp_id = (self._exploration_wp_id + 1) % len(self._search_xy_waypoints)
            self._search_wp_start_tick = self._tick
            target_xy = self._search_xy_waypoints[self._exploration_wp_id].copy()

        direction_xy = target_xy - pos[:2]
        direction_xy = self._add_seen_obstacle_repulsion(direction_xy, pos, estimated_map)
        norm_xy = float(np.linalg.norm(direction_xy))
        if norm_xy > 1e-6:
            direction_xy /= norm_xy
        else:
            direction_xy = np.zeros(2, dtype=float)

        horizontal_gain = float(
            np.clip(
                (pos[2] - self.search_horizontal_start_alt)
                / max(self.search_horizontal_ramp, 1e-3),
                0.0,
                1.0,
            )
        )
        step_xy = self.search_step_xy if target_gate == 0 else self.followup_search_step_xy
        speed_xy = self.search_speed_xy if target_gate == 0 else self.followup_search_speed_xy

        des_pos = pos.copy()
        des_pos[:2] += horizontal_gain * step_xy * direction_xy
        des_pos[2] = target_altitude
        des_vel = np.array(
            [
                horizontal_gain * speed_xy * direction_xy[0],
                horizontal_gain * speed_xy * direction_xy[1],
                np.clip(
                    target_altitude - pos[2],
                    -self.search_climb_speed,
                    self.search_climb_speed,
                ),
            ],
            dtype=float,
        )
        yaw = (
            float(math.atan2(direction_xy[1], direction_xy[0]))
            if norm_xy > 1e-6
            else float(self_state.get("yaw_world", 0.0))
        )

        return {
            "pos": des_pos,
            "vel": des_vel,
            "acc": np.zeros(3),
            "yaw": yaw,
            "omega": np.zeros(3),
            "wp_id": self._exploration_wp_id,
            "wp_type": f"g{target_gate}_route_search",
            "gate_id": -1,
        }

    def _gate_staging_reference(
        self,
        self_state: dict,
        planning_map: dict,
        target_gate: int,
    ) -> dict | None:
        """Move to a gate-centerline staging point before committing to traversal."""
        if not self._is_random_track or self._use_fixed_seed_map:
            return None
        if self._staged_target_gate == target_gate:
            return None

        gate = planning_map["gates"][0]
        gate_pos = np.asarray(gate["pos_world"], dtype=float)
        normal = np.asarray(gate.get("normal", [1.0, 0.0, 0.0]), dtype=float)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-6:
            return None
        normal /= normal_norm

        pos = np.asarray(self_state["pos_world"], dtype=float)
        lateral_axis = np.array([-normal[1], normal[0], 0.0], dtype=float)
        lateral_norm = float(np.linalg.norm(lateral_axis))
        if lateral_norm > 1e-6:
            lateral_axis /= lateral_norm

        staging_distance = 0.78 if target_gate == 0 else 0.68
        staging_pos = gate_pos - staging_distance * normal
        staging_pos[2] = float(np.clip(gate_pos[2], 0.48, 1.24))

        rel = pos - gate_pos
        plane_progress = float(np.dot(rel, normal))
        lateral_error = abs(float(np.dot(rel, lateral_axis))) if lateral_norm > 1e-6 else 0.0
        vertical_error = abs(float(rel[2]))
        staging_error = float(np.linalg.norm(pos - staging_pos))

        if self._search_target_gate != target_gate:
            self._staging_start_tick = self._tick

        timed_out = (
            self._tick - self._staging_start_tick
            > int(2.2 * self._freq)
        )
        staged = (
            staging_error < 0.26
            or (
                plane_progress < -0.34
                and lateral_error < 0.20
                and vertical_error < 0.16
            )
            or (
                timed_out
                and plane_progress < -0.25
                and lateral_error < 0.28
                and vertical_error < 0.22
            )
        )
        if staged:
            self._staged_target_gate = target_gate
            self.i_error[:] = 0.0
            self.trajectory_manager.initialized = False
            return None

        delta = staging_pos - pos
        distance = float(np.linalg.norm(delta))
        if distance > 1e-6:
            direction = delta / distance
        else:
            direction = np.zeros(3, dtype=float)
        speed = 0.18 if distance < 0.55 else 0.24
        des_vel = speed * direction
        des_vel[2] = float(np.clip(staging_pos[2] - pos[2], -0.16, 0.16))
        yaw = float(math.atan2(normal[1], normal[0]))

        return {
            "pos": staging_pos,
            "vel": des_vel,
            "acc": np.zeros(3),
            "yaw": yaw,
            "omega": np.zeros(3),
            "wp_id": -100 - int(target_gate),
            "wp_type": f"g{target_gate}_gate_staging",
            "gate_id": target_gate,
        }

    def _search_altitude_for_gate(self, target_gate: int) -> float:
        """Choose a safe search altitude near the nominal height of target_gate."""
        if target_gate == 0:
            return self.first_gate_search_altitude
        nominal_gates = getattr(self.raw_observation, "nominal_gates", [])
        gate_height = 1.0
        if 0 <= target_gate < len(nominal_gates):
            gate_height = float(np.asarray(nominal_gates[target_gate]["pos"], dtype=float)[2])
        return float(
            np.clip(
                gate_height + self.search_gate_clearance,
                self.search_min_altitude,
                self.search_max_altitude,
            )
        )

    def _add_seen_obstacle_repulsion(
        self,
        direction_xy: NDArray[np.floating],
        pos: NDArray[np.floating],
        estimated_map: dict,
    ) -> NDArray[np.floating]:
        """Bias search motion away from already observed obstacles."""
        adjusted = np.asarray(direction_xy, dtype=float).copy()
        for obstacle in estimated_map.get("obstacles", []):
            if not obstacle.get("seen", False):
                continue
            offset = pos[:2] - np.asarray(obstacle["pos_world"][:2], dtype=float)
            xy_dist = float(np.linalg.norm(offset))
            influence_radius = self.trajectory_manager.obstacle_safety_radius + 0.34
            if xy_dist < influence_radius:
                adjusted += 1.5 * (influence_radius - xy_dist) * offset / (xy_dist + 1e-6)
        return adjusted

    def _race_finished(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Return True once the simulator reports all gates as passed."""
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
            self._reset_reference_smoothing()
            self._staged_target_gate = None
            self._staging_start_tick = self._tick
            self._set_waypoint_to_gate(current_target_gate)

        self._last_target_gate = current_target_gate

    def _set_waypoint_to_gate(self, gate_id: int) -> None:
        """Move TrajectoryManager to the first waypoint of gate_id.

        The planner may insert extra align/detour points, so do not assume
        a fixed number of waypoints per gate.
        """
        if not self.trajectory_manager.waypoints:
            return

        for i, wp in enumerate(self.trajectory_manager.waypoints):
            if int(wp.get("gate_id", -1)) == int(gate_id):
                self.trajectory_manager.current_wp_id = i
                return

        self.trajectory_manager.current_wp_id = int(
            np.clip(
                self.trajectory_manager.current_wp_id,
                0,
                len(self.trajectory_manager.waypoints) - 1,
            )
        )

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

    def _smooth_reference(
        self,
        ref: dict,
        des_pos: NDArray[np.floating],
        des_vel: NDArray[np.floating],
        current_pos: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Low-pass reference jumps that otherwise create lateral oscillation."""
        ref_key = (ref.get("wp_id"), ref.get("wp_type"), ref.get("gate_id"))
        wp_type = str(ref.get("wp_type", ""))
        precision_types = {
            "entry",
            "align",
            "center",
            "previous_gate_recross",
            "dynamic_probe",
        }
        is_precision = wp_type in precision_types or wp_type.endswith("_gate_staging")

        if (
            self._smoothed_ref_pos is None
            or self._smoothed_ref_vel is None
            or ref_key != self._last_ref_key
            or is_precision
        ):
            self._smoothed_ref_pos = des_pos.copy()
            self._smoothed_ref_vel = des_vel.copy()
            self._last_ref_key = ref_key
            return des_pos, des_vel

        alpha_pos = self.ref_pos_smoothing_alpha
        alpha_vel = self.ref_vel_smoothing_alpha
        if wp_type in {"obstacle_detour", "gate_frame_detour", "dynamic_replan"}:
            alpha_pos *= 0.82
            alpha_vel *= 0.82

        self._smoothed_ref_pos = (
            (1.0 - alpha_pos) * self._smoothed_ref_pos
            + alpha_pos * des_pos
        )
        self._smoothed_ref_vel = (
            (1.0 - alpha_vel) * self._smoothed_ref_vel
            + alpha_vel * des_vel
        )

        raw_error = des_pos - current_pos
        smooth_error = self._smoothed_ref_pos - current_pos
        if np.linalg.norm(smooth_error) > np.linalg.norm(raw_error) + 0.18:
            self._smoothed_ref_pos = des_pos.copy()

        return self._smoothed_ref_pos.copy(), self._smoothed_ref_vel.copy()

    def _limit_action_rate(
        self,
        action: NDArray[np.floating],
        ref: dict,
    ) -> NDArray[np.floating]:
        """Limit command slew rate to reduce avoidable roll/pitch chatter."""
        limited = np.asarray(action, dtype=float).copy()
        previous = np.asarray(self._previous_action, dtype=float)
        wp_type = str(ref.get("wp_type", ""))

        attitude_step = self.max_attitude_step
        thrust_step = self.max_thrust_step
        if wp_type in {"entry", "align", "center", "previous_gate_recross"}:
            attitude_step *= 1.35
            thrust_step *= 1.35
        elif wp_type in {"obstacle_detour", "gate_frame_detour", "dynamic_replan"}:
            attitude_step *= 0.85
            thrust_step *= 0.90

        max_step = np.array(
            [attitude_step, attitude_step, self.max_yaw_step, thrust_step],
            dtype=float,
        )
        limited = previous + np.clip(limited - previous, -max_step, max_step)
        limited[0] = np.clip(limited[0], -self.tilt_limit_rad, self.tilt_limit_rad)
        limited[1] = np.clip(limited[1], -self.tilt_limit_rad, self.tilt_limit_rad)
        limited[3] = np.clip(limited[3], self.thrust_min, self.thrust_max)
        return limited.astype(np.float32)

    def _reset_reference_smoothing(self) -> None:
        """Clear smoothed reference state after gate changes or episode reset."""
        self._smoothed_ref_pos = None
        self._smoothed_ref_vel = None
        self._last_ref_key = None

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
        """Update post-step controller bookkeeping."""
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
        """Persist debug path data and reset episode-local state."""
        if self.debug and self._actual_path_history:
            try:
                with open("actual_path_history.txt", "w") as f:
                    for pos in self._actual_path_history:
                        f.write(
                            ",".join(map(str, np.round(pos, 4).tolist())) + "\n"
                        )
                print(
                    "[ACTUAL_PATH] saved "
                    f"{len(self._actual_path_history)} points to actual_path_history.txt"
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
        self.trajectory_manager._last_target_gate = None
        self.trajectory_manager._last_dynamic_replan_tick = -100
        self.trajectory_manager._update_tick = 0
        self.trajectory_manager._slowdown_until_tick = -1
        self.trajectory_manager._prev_gate_recross_active = False
        self.trajectory_manager._last_speed_scale = 1.0
        self._last_map_signature = None
        self._last_ref = None
        self._last_waypoints = np.empty((0, 3), dtype=float)
        self._last_target_gate = 0
        self._last_velocity = np.zeros(3, dtype=float)
        self._estimated_acc_world[:] = 0.0
        self._reset_reference_smoothing()
        self._previous_action = np.array(
            [0.0, 0.0, 0.0, self.drone_mass * self.g],
            dtype=np.float32,
        )
        self._exploration_wp_id = None
        self._search_target_gate = None
        self._search_wp_start_tick = 0
        self._staged_target_gate = None
        self._staging_start_tick = 0

    def render_callback(self, sim: Sim):
        """Render the latest reference, waypoint path, and actual path."""
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
