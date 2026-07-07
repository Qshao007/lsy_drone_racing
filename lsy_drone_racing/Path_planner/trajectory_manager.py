"""Obstacle-aware waypoint manager for drone racing.

TrajectoryManager converts the current estimated map into a safe waypoint list.
It protects gate traversal with entry/align/center/exit points, inserts detours
around obstacles and gate frames, prunes redundant points, and schedules a
reference velocity for the attitude controller.
"""

import numpy as np


class TrajectoryManager:
    """Build, repair, and track gate waypoints for the controller."""

    def __init__(
        self,
        waypoint_threshold=0.20,
        entry_distance=0.42,
        exit_distance=0.58,
        max_speed=0.25,
        max_segment_length=0.45,
        obstacle_safety_radius=0.36,
        obstacle_detour_margin=0.08,
        gate_align_distance=0.18,
        gate_pass_speed_scale=0.58,
        enable_dynamic_replan=True,
        enable_previous_gate_recross=True,
        previous_gate_recross_until_gate=None,
        aggressive_speed=False,
        loop=True,
        debug=False,
    ):
        """Initialize geometry, safety, speed, and progress-tracking settings."""
        self.waypoint_threshold = waypoint_threshold
        self.entry_distance = entry_distance
        self.exit_distance = exit_distance
        self.max_speed = max_speed
        self.max_segment_length = max_segment_length
        self.gate_align_distance = gate_align_distance
        self.gate_pass_speed_scale = gate_pass_speed_scale
        self.enable_dynamic_replan = enable_dynamic_replan
        self.enable_previous_gate_recross = enable_previous_gate_recross
        self.previous_gate_recross_until_gate = previous_gate_recross_until_gate
        self.aggressive_speed = aggressive_speed
        self.loop = loop
        self.debug = debug

        self.waypoints = []
        self.current_wp_id = 0
        self.initialized = False
        self._last_target_gate = None
        self._last_dynamic_replan_tick = -100
        self._update_tick = 0
        self._slowdown_until_tick = -1
        self._prev_gate_recross_active = False
        self._clearance_debug_prints = 0
        self._prune_debug_prints = 0
        self._progress_debug_prints = 0
        self._last_speed_scale = 1.0
        self._risk_speed_debug_prints = 0

        # Gate geometry from the challenge description. We treat the four
        # frame bars as a rectangular ring obstacle around the central opening.
        self.gate_opening_half_width = 0.20
        self.gate_opening_half_height = 0.20
        self.gate_outer_half_width = 0.36
        self.gate_outer_half_height = 0.36
        self.gate_frame_clearance = 0.14
        self.gate_frame_depth = 0.40
        # Wooden poles are thin in the map, but the drone has body size and
        # tracking overshoot. Use a larger XY clearance when circling poles.
        self.obstacle_safety_radius = obstacle_safety_radius
        self.obstacle_detour_margin = obstacle_detour_margin

    def build_waypoints_from_map(self, estimated_map, keep_progress=False):
        """Create a full gate route from the latest map estimate."""
        old_wp_id = self.current_wp_id
        waypoints = []

        for gate in estimated_map["gates"]:
            gate_pos = np.asarray(gate["pos_world"], dtype=float)
            gate_yaw = float(gate["yaw"])
            gate_id = int(gate["id"])

            normal = gate.get(
                "normal",
                np.array(
                    [
                        np.cos(gate_yaw),
                        np.sin(gate_yaw),
                        0.0,
                    ],
                    dtype=float,
                ),
            )
            normal = np.asarray(normal, dtype=float)
            normal = normal / (np.linalg.norm(normal) + 1e-9)

            entry = gate_pos - self.entry_distance * normal
            # Align is a short centerline staging waypoint before the gate
            # plane; it improves traversal without hard-coding a world pose.
            align = gate_pos - self.gate_align_distance * normal
            center = gate_pos.copy()
            exit_ = gate_pos + self.exit_distance * normal

            waypoints.append(
                {
                    "type": "entry",
                    "gate_id": gate_id,
                    "pos": entry.copy(),
                    "yaw": gate_yaw,
                }
            )
            waypoints.append(
                {
                    "type": "align",
                    "gate_id": gate_id,
                    "pos": align.copy(),
                    "yaw": gate_yaw,
                }
            )
            waypoints.append(
                {
                    "type": "center",
                    "gate_id": gate_id,
                    "pos": center.copy(),
                    "yaw": gate_yaw,
                }
            )

            waypoints.append(
                {
                    "type": "exit",
                    "gate_id": gate_id,
                    "pos": exit_.copy(),
                    "yaw": gate_yaw,
                }
            )

        # Route construction pipeline:
        # 1) Insert obstacle/frame detours.
        # 2) Push non-critical points away from poles.
        # 3) Remove redundant points only when direct segments remain safe.
        # 4) Densify long remaining segments for smoother tracking.
        waypoints = self._avoid_obstacles(waypoints, estimated_map["obstacles"])
        waypoints = self._avoid_gate_frames(waypoints, estimated_map["gates"])
        waypoints = self._push_waypoints_away_from_obstacles(waypoints, estimated_map["obstacles"])
        waypoints = self._avoid_obstacles(waypoints, estimated_map["obstacles"])
        waypoints = self._shortcut_clear_waypoints(waypoints, estimated_map)
        waypoints = self._prune_redundant_waypoints(waypoints, estimated_map)
        waypoints = self._densify_waypoints(waypoints)
        self.waypoints = waypoints

        if keep_progress and len(self.waypoints) > 0:
            self.current_wp_id = min(old_wp_id, len(self.waypoints) - 1)
        else:
            self.current_wp_id = 0

        self.initialized = True

        if self.debug:
            self.print_waypoints(estimated_map)

    def update(self, self_state, estimated_map, target_gate=None):
        """Return the current reference position, velocity, and waypoint metadata."""
        self._update_tick += 1

        if not self.initialized:
            self.build_waypoints_from_map(estimated_map)
            self._prepend_current_segment_detours(self_state, estimated_map)

        if len(self.waypoints) == 0:
            return self._hover_reference(self_state)

        just_synced = False

        if target_gate is not None:
            just_synced = self.sync_to_target_gate(
                int(target_gate),
                estimated_map,
            )

        drone_pos = np.asarray(self_state["pos_world"], dtype=float)

        if target_gate is not None:
            self._guard_gate_axis_progress(
                int(target_gate),
                drone_pos,
                estimated_map,
            )

        current_wp = self.waypoints[self.current_wp_id]
        target_pos = current_wp["pos"]

        distance = np.linalg.norm(target_pos - drone_pos)

        if not just_synced:
            while distance < self._threshold_for_waypoint(current_wp):
                old_wp_id = self.current_wp_id
                self._advance_waypoint()

                if self.current_wp_id == old_wp_id:
                    break

                current_wp = self.waypoints[self.current_wp_id]
                target_pos = current_wp["pos"]
                distance = np.linalg.norm(target_pos - drone_pos)

        current_wp = self._repair_current_segment_if_needed(
            self_state,
            estimated_map,
            current_wp,
        )
        if target_gate is not None:
            current_wp = self._previous_gate_recross_reference(
                int(target_gate),
                drone_pos,
                estimated_map,
                current_wp,
            )
        target_pos = current_wp["pos"]
        distance = np.linalg.norm(target_pos - drone_pos)

        progress_target = self._progress_reference_target(
            drone_pos,
            current_wp,
            estimated_map,
            int(target_gate) if target_gate is not None else None,
        )
        if progress_target is not None:
            target_pos = progress_target
            distance = np.linalg.norm(target_pos - drone_pos)

        ref_vel = self._compute_reference_velocity(
            drone_pos,
            target_pos,
            current_wp,
            estimated_map,
            int(target_gate) if target_gate is not None else None,
        )

        if self.debug and self._update_tick % 10 == 0:
            print(
                f"[TRACK] "
                f"target_gate={target_gate} "
                f"wp_id={self.current_wp_id} "
                f"gate={current_wp['gate_id']} "
                f"type={current_wp['type']} "
                f"dist={distance:.2f}"
            )
            print(
                "drone =",
                np.round(drone_pos, 2),
                "target =",
                np.round(target_pos, 2),
            )

        return {
            "pos": target_pos.copy(),
            "vel": ref_vel,
            "acc": np.zeros(3),
            "yaw": current_wp["yaw"],
            "omega": np.zeros(3),
            "wp_id": self.current_wp_id,
            "wp_type": current_wp["type"],
            "gate_id": current_wp["gate_id"],
        }

    def sync_to_target_gate(self, target_gate, estimated_map):
        """Trust simulator gate progress when it has advanced past our waypoint index."""
        if target_gate < 0:
            return False

        if self._last_target_gate is None:
            self._last_target_gate = target_gate
            self._jump_to_gate_start(target_gate)
            return True

        if target_gate != self._last_target_gate:
            self.build_waypoints_from_map(
                estimated_map,
                keep_progress=False,
            )

            if target_gate > 0:
                self._jump_to_gate_exit(target_gate - 1)
            else:
                self._jump_to_gate_start(target_gate)
            self._last_target_gate = target_gate
            if self.debug:
                print(
                    "[SYNC_TARGET_CHANGE]",
                    "target_gate=",
                    target_gate,
                    "current_wp_id=",
                    self.current_wp_id,
                    "current_wp=",
                    self.waypoints[self.current_wp_id]["type"],
                    "gate=",
                    self.waypoints[self.current_wp_id]["gate_id"],
                )
            return True

        current_wp = self.waypoints[self.current_wp_id]

        if (
            target_gate > 0
            and current_wp["type"] == "exit"
            and int(current_wp["gate_id"]) == target_gate - 1
        ):
            return False

        if current_wp["gate_id"] < target_gate:
            self._jump_to_gate_start(target_gate)
            return True

        return False

    def _jump_to_gate_start(self, gate_id):
        for i, wp in enumerate(self.waypoints):
            if wp["gate_id"] == gate_id:
                self.current_wp_id = i

                if self.debug:
                    print(
                        "[SYNC]",
                        "target_gate=",
                        gate_id,
                        "jump_to_wp=",
                        i,
                        "type=",
                        wp["type"],
                    )

                return

        if len(self.waypoints) > 0:
            self.current_wp_id = len(self.waypoints) - 1

    def _jump_to_gate_exit(self, gate_id):
        for i, wp in enumerate(self.waypoints):
            if int(wp["gate_id"]) == int(gate_id) and wp["type"] == "exit":
                self.current_wp_id = i

                if self.debug:
                    print(
                        "[SYNC]",
                        "target_previous_exit=",
                        gate_id,
                        "jump_to_wp=",
                        i,
                    )

                return

        self._jump_to_gate_start(gate_id + 1)

    def update_map_if_needed(self, estimated_map):
        """Rebuild waypoints while preserving progress after map changes."""
        self.build_waypoints_from_map(
            estimated_map,
            keep_progress=True,
        )

    def _prepend_current_segment_detours(self, self_state, estimated_map):
        """Repair the initial segment from drone position to first waypoint."""
        if len(self.waypoints) == 0:
            return

        start_pos = np.asarray(self_state["pos_world"], dtype=float)
        current_wp = self.waypoints[self.current_wp_id]
        target_pos = np.asarray(current_wp["pos"], dtype=float)

        if not self._segment_needs_replan(start_pos, target_pos, current_wp, estimated_map):
            return

        start_wp = {
            "type": "dynamic_probe",
            "gate_id": int(current_wp["gate_id"]),
            "pos": start_pos.copy(),
            "yaw": current_wp["yaw"],
        }
        local = [start_wp, current_wp]
        local = self._avoid_obstacles(local, estimated_map["obstacles"])
        local = self._avoid_gate_frames(local, estimated_map["gates"])
        local = self._push_waypoints_away_from_obstacles(local, estimated_map["obstacles"])
        local = self._avoid_obstacles(local, estimated_map["obstacles"])
        local = self._shortcut_clear_waypoints(local, estimated_map)
        local = self._prune_redundant_waypoints(local, estimated_map)
        local = self._densify_waypoints(local)

        direction = target_pos - start_pos
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            return

        target_dir = direction / direction_norm
        inserted = []
        for wp in local[1:-1]:
            step = np.asarray(wp["pos"], dtype=float) - start_pos
            if np.dot(step, target_dir) < 0.04:
                continue

            new_wp = dict(wp)
            new_wp["type"] = "dynamic_replan"
            inserted.append(new_wp)

        if not inserted:
            return

        self.waypoints[self.current_wp_id:self.current_wp_id] = inserted

        if self.debug:
            print(
                "[INIT_SEGMENT_INSERT]",
                "inserted=",
                [np.round(wp["pos"], 2).tolist() for wp in inserted],
            )

    def _repair_current_segment_if_needed(self, self_state, estimated_map, current_wp):
        """Insert local detours if a newly observed object blocks the current segment."""
        if not self.enable_dynamic_replan:
            return current_wp

        drone_pos = np.asarray(self_state["pos_world"], dtype=float)
        target_pos = np.asarray(current_wp["pos"], dtype=float)

        segment_needs_replan = self._segment_needs_replan(
            drone_pos,
            target_pos,
            current_wp,
            estimated_map,
        )
        if segment_needs_replan:
            self._slowdown_until_tick = max(self._slowdown_until_tick, self._update_tick + 8)

        replan_safe_types = {
            "entry",
            "align",
            "intermediate",
            "obstacle_detour",
            "gate_frame_detour",
        }
        if current_wp["type"] not in replan_safe_types:
            return current_wp

        if int(current_wp["gate_id"]) <= 0 and current_wp["type"] != "entry":
            return current_wp

        if not segment_needs_replan:
            return current_wp

        if self._update_tick - self._last_dynamic_replan_tick < 14:
            return current_wp

        new_waypoints = self._make_dynamic_replan_waypoints(
            self_state,
            estimated_map,
            current_wp,
        )

        if len(new_waypoints) == 0:
            self._last_dynamic_replan_tick = self._update_tick
            if self.debug:
                print(
                    "[REPLAN_REJECT]",
                    "tick=",
                    self._update_tick,
                    "wp_type=",
                    current_wp["type"],
                    "gate=",
                    current_wp["gate_id"],
                    "reason=no_clear_candidate",
                )
            return current_wp

        while (
            self.current_wp_id < len(self.waypoints)
            and self.waypoints[self.current_wp_id]["type"] == "dynamic_replan"
        ):
            del self.waypoints[self.current_wp_id]

        self.waypoints[self.current_wp_id:self.current_wp_id] = new_waypoints
        self._last_dynamic_replan_tick = self._update_tick
        if self.debug:
            print(
                "[REPLAN_INSERT]",
                "tick=",
                self._update_tick,
                "before_type=",
                current_wp["type"],
                "gate=",
                current_wp["gate_id"],
                "inserted=",
                [np.round(wp["pos"], 2).tolist() for wp in new_waypoints],
            )
        return self.waypoints[self.current_wp_id]

    def _guard_gate_axis_progress(self, target_gate, drone_pos, estimated_map):
        """Keep entry/align/center ordering until the drone crosses the gate plane."""
        if not self.waypoints:
            return

        gate = self._find_gate_by_id(estimated_map, target_gate)
        if gate is None:
            return

        gate_pos = np.asarray(gate["pos_world"], dtype=float)

        normal = np.asarray(
            gate.get(
                "normal",
                np.array(
                    [
                        np.cos(float(gate["yaw"])),
                        np.sin(float(gate["yaw"])),
                        0.0,
                    ],
                    dtype=float,
                ),
            ),
            dtype=float,
        )
        normal = normal / (np.linalg.norm(normal) + 1e-9)

        rel = np.asarray(drone_pos, dtype=float) - gate_pos
        signed_progress = float(np.dot(rel, normal))

        lateral_axis = np.array([-normal[1], normal[0], 0.0], dtype=float)
        lateral_axis = lateral_axis / (np.linalg.norm(lateral_axis) + 1e-9)

        lateral_error = float(np.dot(rel, lateral_axis))
        vertical_error = float(rel[2])

        entry_id = self._find_waypoint_id(target_gate, "entry")
        align_id = self._find_waypoint_id(target_gate, "align")
        center_id = self._find_waypoint_id(target_gate, "center")

        if align_id is None or center_id is None:
            return

        # Do not let this guard interfere with approach/detour waypoints
        # before the current gate entry. This prevents the gate-1 long-jump bug.
        if entry_id is not None and self.current_wp_id < entry_id:
            return

        current_wp = self.waypoints[self.current_wp_id]
        current_type = current_wp.get("type")
        current_gate = int(current_wp.get("gate_id", -1))

        still_before_gate = signed_progress < 0.03
        after_center_types = {
            "intermediate",
            "exit",
            "obstacle_detour",
            "gate_frame_detour",
        }

        if (
            current_gate == int(target_gate)
            and still_before_gate
            and current_type in after_center_types
        ):
            if signed_progress < -0.12:
                self.current_wp_id = align_id
            else:
                self.current_wp_id = center_id

            self._slowdown_until_tick = max(
                self._slowdown_until_tick,
                self._update_tick + 30,
            )

            if self.debug:
                print(
                    "[GATE_GUARD_RESET]",
                    "gate=", target_gate,
                    "from_type=", current_type,
                    "to_wp=", self.current_wp_id,
                    "signed_progress=", round(signed_progress, 3),
                    "lat_err=", round(lateral_error, 3),
                    "z_err=", round(vertical_error, 3),
                )

            return

        near_gate_plane = -0.18 <= signed_progress <= 0.08
        not_centered = abs(lateral_error) > 0.075 or abs(vertical_error) > 0.075

        if current_gate == int(target_gate) and near_gate_plane and not_centered:
            self.current_wp_id = center_id
            self._slowdown_until_tick = max(
                self._slowdown_until_tick,
                self._update_tick + 35,
            )

            if self.debug:
                print(
                    "[GATE_GUARD_CENTER]",
                    "gate=", target_gate,
                    "signed_progress=", round(signed_progress, 3),
                    "lat_err=", round(lateral_error, 3),
                    "z_err=", round(vertical_error, 3),
                )


    def _previous_gate_recross_reference(
        self,
        target_gate,
        drone_pos,
        estimated_map,
        current_wp,
    ):
        """Route old-gate recrosses through the opening corridor, not the frame."""
        recross_allowed = self.enable_previous_gate_recross and (
            self.previous_gate_recross_until_gate is None
            or target_gate < self.previous_gate_recross_until_gate
        )
        if not recross_allowed or target_gate <= 0 or not self.waypoints:
            self._prev_gate_recross_active = False
            return current_wp

        previous_gate_id = target_gate - 1
        previous_exit_id = self._find_waypoint_id(previous_gate_id, "exit")
        current_entry_id = self._find_waypoint_id(target_gate, "entry")
        if previous_exit_id is None or current_entry_id is None:
            self._prev_gate_recross_active = False
            return current_wp

        if not (previous_exit_id <= self.current_wp_id < current_entry_id):
            self._prev_gate_recross_active = False
            return current_wp

        gate = self._find_gate_by_id(estimated_map, previous_gate_id)
        if gate is None:
            self._prev_gate_recross_active = False
            return current_wp

        gate_pos, normal, lateral_axis = self._gate_position_normal_lateral(gate)
        rel = np.asarray(drone_pos, dtype=float) - gate_pos
        signed_progress = float(np.dot(rel, normal))
        lateral_error = float(np.dot(rel, lateral_axis))
        vertical_error = float(rel[2])

        release_progress = -0.25
        center_progress = 0.07

        if signed_progress <= release_progress:
            self._prev_gate_recross_active = False
            self._skip_to_previous_gate_front_side_waypoint(
                previous_gate_id,
                target_gate,
                current_entry_id,
                gate_pos,
                normal,
            )
            return self.waypoints[self.current_wp_id]

        if signed_progress > center_progress:
            phase = "center"
            target_pos = gate_pos.copy()
        else:
            phase = "front_align"
            align_distance = max(self.gate_align_distance, 0.28)
            target_pos = gate_pos - align_distance * normal

        self._prev_gate_recross_active = True
        self._slowdown_until_tick = max(self._slowdown_until_tick, self._update_tick + 12)

        if self.debug and self._update_tick % 5 == 0:
            print(
                "[PREV_GATE_RECROSS]",
                "prev_gate=", previous_gate_id,
                "target_gate=", target_gate,
                "phase=", phase,
                "signed=", round(signed_progress, 3),
                "lat=", round(lateral_error, 3),
                "z=", round(vertical_error, 3),
                "target=", np.round(target_pos, 3).tolist(),
            )

        return {
            "type": "previous_gate_recross",
            "gate_id": previous_gate_id,
            "pos": target_pos.copy(),
            "yaw": float(gate["yaw"]),
        }

    def _skip_to_previous_gate_front_side_waypoint(
        self,
        previous_gate_id,
        target_gate,
        current_entry_id,
        gate_pos,
        normal,
    ):
        for wp_id in range(self.current_wp_id, current_entry_id):
            wp = self.waypoints[wp_id]
            if int(wp.get("gate_id", -1)) != int(target_gate):
                continue

            progress = float(np.dot(np.asarray(wp["pos"], dtype=float) - gate_pos, normal))
            if progress <= -0.12:
                if wp_id > self.current_wp_id:
                    self.current_wp_id = wp_id
                    if self.debug:
                        print(
                            "[PREV_GATE_RECROSS]",
                            "prev_gate=", previous_gate_id,
                            "target_gate=", target_gate,
                            "phase=release",
                            "next_wp=", self.current_wp_id,
                            "next_type=", wp["type"],
                        )
                return

        # If all pre-entry detours are behind the old gate plane, resume at
        # the current gate entry instead of pulling the drone back through the
        # old frame a second time.
        if self.current_wp_id < current_entry_id:
            self.current_wp_id = current_entry_id
            if self.debug:
                print(
                    "[PREV_GATE_RECROSS]",
                    "prev_gate=", previous_gate_id,
                    "target_gate=", target_gate,
                    "phase=release",
                    "next_wp=", self.current_wp_id,
                    "next_type=", self.waypoints[self.current_wp_id]["type"],
                )

    def _gate_position_normal_lateral(self, gate):
        """Return gate center, forward normal, and lateral axis."""
        gate_pos = np.asarray(gate["pos_world"], dtype=float)
        normal = np.asarray(gate.get("normal", self._normal_from_yaw(gate["yaw"])), dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        lateral_axis = np.array([-normal[1], normal[0], 0.0], dtype=float)
        lateral_axis = lateral_axis / (np.linalg.norm(lateral_axis) + 1e-9)
        return gate_pos, normal, lateral_axis


    def _find_gate_by_id(self, estimated_map, gate_id):
        """Find one gate record by id."""
        for gate in estimated_map.get("gates", []):
            if int(gate.get("id", -1)) == int(gate_id):
                return gate
        return None


    def _find_waypoint_id(self, gate_id, wp_type):
        """Find the first waypoint matching a gate id and type."""
        for i, wp in enumerate(self.waypoints):
            if int(wp.get("gate_id", -1)) == int(gate_id) and wp.get("type") == wp_type:
                return i
        return None


    def _threshold_for_waypoint(self, waypoint):
        """Choose how close the drone must get before advancing a waypoint."""
        wp_type = waypoint["type"]

        if self.aggressive_speed:
            if wp_type in {"obstacle_detour", "gate_frame_detour", "dynamic_replan"}:
                return max(self.waypoint_threshold, 0.44)

            if wp_type == "intermediate":
                return max(self.waypoint_threshold, 0.40)

            if wp_type == "entry":
                return min(max(self.waypoint_threshold, 0.16), 0.18)

            if wp_type == "align":
                return min(max(self.waypoint_threshold, 0.12), 0.14)

            if wp_type == "center":
                return min(max(self.waypoint_threshold, 0.10), 0.11)

        if wp_type in {"obstacle_detour", "gate_frame_detour", "dynamic_replan"}:
            return max(self.waypoint_threshold, 0.40)

        if wp_type == "intermediate":
            return max(self.waypoint_threshold, 0.34)

        if wp_type == "entry":
            return min(self.waypoint_threshold, 0.12)

        if wp_type == "align":
          return min(self.waypoint_threshold, 0.10)

        if wp_type == "center":
           return min(self.waypoint_threshold, 0.09)

        return self.waypoint_threshold

    def _progress_reference_target(self, drone_pos, current_wp, estimated_map, target_gate):
        """Use a temporary lookahead target on safe open segments without editing waypoints."""
        if not self._can_use_progress_reference(current_wp, target_gate):
            return None

        next_id = self.current_wp_id + 1
        if next_id >= len(self.waypoints):
            return None

        next_wp = self.waypoints[next_id]
        lookahead = self._lookahead_for_waypoint_type(current_wp, next_wp, target_gate)
        if lookahead <= 0.0:
            return None

        segment_start = np.asarray(current_wp["pos"], dtype=float)
        segment_end = np.asarray(next_wp["pos"], dtype=float)
        segment = segment_end - segment_start
        segment_length = float(np.linalg.norm(segment))
        if segment_length < 1e-6:
            return None

        alpha = self._segment_projection_progress(drone_pos, segment_start, segment)
        advanced_alpha = min(1.0, alpha + lookahead / segment_length)
        progress_target = segment_start + advanced_alpha * segment

        if not self._progress_reference_is_safe(
            drone_pos,
            np.asarray(current_wp["pos"], dtype=float),
            progress_target,
            current_wp,
            estimated_map,
        ):
            return None

        if self.debug and self._update_tick % 10 == 0 and self._progress_debug_prints < 12:
            print(
                "[PROGRESS_REF]",
                "wp_id=", self.current_wp_id,
                "type=", current_wp["type"],
                "alpha=", round(alpha, 3),
                "lookahead=", round(lookahead, 3),
                "target=", np.round(progress_target, 3).tolist(),
            )
            self._progress_debug_prints += 1

        return progress_target

    @staticmethod
    def _segment_projection_progress(point, segment_start, segment):
        denom = float(np.dot(segment, segment))
        if denom < 1e-9:
            return 0.0
        alpha = float(np.dot(np.asarray(point, dtype=float) - segment_start, segment) / denom)
        return float(np.clip(alpha, 0.0, 1.0))

    def _lookahead_for_waypoint_type(self, current_wp, next_wp, target_gate):
        current_type = current_wp.get("type")

        if (
            self._is_gate_precision_waypoint(current_wp)
            or self._is_gate_precision_waypoint(next_wp)
        ):
            return 0.0

        if current_type == "exit":
            if target_gate is not None and int(current_wp.get("gate_id", -1)) != int(target_gate):
                return 0.0
            return 0.08 if self.aggressive_speed else 0.04

        if self.aggressive_speed:
            lookahead_by_type = {
                "intermediate": 0.30,
                "obstacle_detour": 0.20,
                "gate_frame_detour": 0.18,
                "dynamic_replan": 0.24,
            }
            return lookahead_by_type.get(current_type, 0.0)

        lookahead_by_type = {
            "intermediate": 0.18,
            "obstacle_detour": 0.12,
            "gate_frame_detour": 0.10,
            "dynamic_replan": 0.14,
        }
        return lookahead_by_type.get(current_type, 0.0)

    def _can_use_progress_reference(self, current_wp, target_gate):
        current_type = current_wp.get("type")
        if current_type in {"align", "center", "previous_gate_recross", "dynamic_probe", "entry"}:
            return False

        if current_type == "exit":
            if target_gate is None:
                return True
            return int(current_wp.get("gate_id", -1)) == int(target_gate)

        allowed_types = {
            "intermediate",
            "obstacle_detour",
            "gate_frame_detour",
            "dynamic_replan",
        }
        if current_type not in allowed_types:
            return False

        if target_gate is not None and int(current_wp.get("gate_id", -1)) != int(target_gate):
            return False

        return True

    @staticmethod
    def _is_gate_precision_waypoint(waypoint):
        return waypoint.get("type") in {
            "entry",
            "align",
            "center",
            "previous_gate_recross",
            "dynamic_probe",
        }

    def _progress_reference_is_safe(
        self,
        drone_pos,
        original_target,
        progress_target,
        current_wp,
        estimated_map,
    ):
        temp_wp = {
            "type": current_wp.get("type", "progress_ref"),
            "gate_id": int(current_wp.get("gate_id", -1)),
            "pos": np.asarray(progress_target, dtype=float),
            "yaw": current_wp.get("yaw", 0.0),
        }
        if self._segment_needs_replan(
            np.asarray(drone_pos, dtype=float),
            np.asarray(progress_target, dtype=float),
            temp_wp,
            estimated_map,
        ):
            return False

        obstacles = estimated_map.get("obstacles", [])
        gates = estimated_map.get("gates", [])
        original_clearance = self._min_obstacle_clearance([original_target], obstacles)
        progress_clearance = self._min_obstacle_clearance([progress_target], obstacles)
        if progress_clearance + 0.01 < original_clearance:
            return False

        original_risk = self._route_gate_frame_risk([drone_pos, original_target], gates)
        progress_risk = self._route_gate_frame_risk([drone_pos, progress_target], gates)
        return progress_risk <= original_risk + 0.03

    def _segment_needs_replan(self, p0, p1, wp1, estimated_map):
        """Return True if direct flight from p0 to p1 crosses known risk."""
        if np.linalg.norm(p1 - p0) < 1e-6:
            return False

        tmp0 = {
            "type": "dynamic_probe",
            "gate_id": int(wp1["gate_id"]),
            "pos": np.asarray(p0, dtype=float),
            "yaw": wp1["yaw"],
        }

        obstacle_detours = self._obstacle_detours(tmp0, wp1, estimated_map["obstacles"])
        if obstacle_detours:
            return True

        gate_detour = self._gate_frame_detour(tmp0, wp1, estimated_map["gates"])
        return gate_detour is not None

    def _segment_hits_obstacle(self, p0, p1, wp1, estimated_map):
        if np.linalg.norm(p1 - p0) < 1e-6:
            return False

        tmp0 = {
            "type": "dynamic_probe",
            "gate_id": int(wp1["gate_id"]),
            "pos": np.asarray(p0, dtype=float),
            "yaw": wp1["yaw"],
        }

        return bool(self._obstacle_detours(tmp0, wp1, estimated_map["obstacles"]))

    def _make_dynamic_replan_waypoints(self, self_state, estimated_map, current_wp):
        """Generate candidate local detours from the current drone state."""
        pos = np.asarray(self_state["pos_world"], dtype=float)
        vel = np.asarray(self_state.get("vel_world", np.zeros(3)), dtype=float)
        acc = np.asarray(self_state.get("acc_world", np.zeros(3)), dtype=float)

        target = np.asarray(current_wp["pos"], dtype=float)
        direction = target - pos
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            return []

        target_dir = direction / direction_norm
        lateral_dir = np.array([-target_dir[1], target_dir[0], 0.0], dtype=float)
        lateral_norm = np.linalg.norm(lateral_dir)
        if lateral_norm < 1e-6:
            lateral_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            lateral_dir = lateral_dir / lateral_norm

        candidates = []
        for horizon in (0.15, 0.25, 0.35, 0.50, 0.70, 0.90):
            predicted_step = horizon * vel + 0.5 * horizon * horizon * acc
            forward_step = float(np.clip(np.dot(predicted_step, target_dir), 0.16, 0.60))
            lateral_step = predicted_step - np.dot(predicted_step, target_dir) * target_dir
            lateral_step[2] = 0.0
            lateral_step_norm = np.linalg.norm(lateral_step)
            if lateral_step_norm > 0.12:
                lateral_step = lateral_step / lateral_step_norm * 0.12

            for lateral_bias in (
                0.0,
                0.18,
                -0.18,
                0.32,
                -0.32,
                0.48,
                -0.48,
                0.64,
                -0.64,
                0.80,
                -0.80,
                1.00,
                -1.00,
            ):
                transition = (
                    pos
                    + forward_step * target_dir
                    + lateral_step
                    + lateral_bias * lateral_dir
                )

                transition_wp = {
                    "type": "dynamic_replan",
                    "gate_id": int(current_wp["gate_id"]),
                    "pos": transition.copy(),
                    "yaw": current_wp["yaw"],
                }
                transition_wp = self._push_single_waypoint_away_from_obstacles(
                    transition_wp,
                    estimated_map["obstacles"],
                )

                local = [transition_wp, current_wp]
                local = self._avoid_obstacles(local, estimated_map["obstacles"])
                local = self._avoid_gate_frames(local, estimated_map["gates"])
                local = self._prune_redundant_waypoints(local, estimated_map)

                repaired = []
                valid_forward = True
                for wp in local[:-1]:
                    step = np.asarray(wp["pos"], dtype=float) - pos
                    if np.dot(step, target_dir) < 0.06:
                        valid_forward = False
                        break

                    new_wp = dict(wp)
                    new_wp["type"] = "dynamic_replan"
                    repaired.append(new_wp)

                if not valid_forward or len(repaired) == 0:
                    continue

                repaired = repaired[:3]
                if not self._route_is_clear(pos, repaired, current_wp, estimated_map):
                    continue

                score = self._score_detour_route(
                    pos,
                    target,
                    [np.asarray(wp["pos"], dtype=float) for wp in repaired],
                    obstacles=estimated_map["obstacles"],
                    gates=estimated_map["gates"],
                    label="dynamic",
                )
                candidates.append((score, repaired))

        if not candidates:
            if self.debug:
                n_horizons = 6
                n_biases = 13
                print(
                    "[TrajectoryManager] no_clear_candidate: dynamic replan tried"
                    f" {n_horizons * n_biases} candidates"
                )
            return []

        return max(candidates, key=lambda item: item[0])[1]

    def _route_is_clear(self, start_pos, inserted_wps, final_wp, estimated_map):
        prev_wp = {
            "type": "dynamic_probe",
            "gate_id": int(final_wp["gate_id"]),
            "pos": np.asarray(start_pos, dtype=float),
            "yaw": final_wp["yaw"],
        }
        route = inserted_wps + [final_wp]

        for wp in route:
            if self._segment_needs_replan(
                np.asarray(prev_wp["pos"], dtype=float),
                np.asarray(wp["pos"], dtype=float),
                wp,
                estimated_map,
            ):
                return False
            prev_wp = wp

        return True

    def _push_single_waypoint_away_from_obstacles(self, waypoint, obstacles):
        pushed = self._push_waypoints_away_from_obstacles([waypoint], obstacles)
        return pushed[0]

    def _advance_waypoint(self):
        """Move waypoint index forward, looping or clamping at the end."""
        self.current_wp_id += 1

        if self.current_wp_id >= len(self.waypoints):
            if self.loop:
                self.current_wp_id = 0
            else:
                self.current_wp_id = len(self.waypoints) - 1

    def _densify_waypoints(self, waypoints):
        """Split long segments into intermediate points for smoother tracking."""
        if len(waypoints) <= 1:
            return waypoints

        dense = [waypoints[0]]

        for i in range(len(waypoints) - 1):
            wp0 = waypoints[i]
            wp1 = waypoints[i + 1]

            p0 = np.asarray(wp0["pos"], dtype=float)
            p1 = np.asarray(wp1["pos"], dtype=float)

            dist = np.linalg.norm(p1 - p0)
            n_insert = int(np.floor(dist / self.max_segment_length))

            for k in range(1, n_insert + 1):
                alpha = k / (n_insert + 1)
                pos = (1.0 - alpha) * p0 + alpha * p1

                dense.append(
                    {
                        "type": "intermediate",
                        "gate_id": wp1["gate_id"],
                        "pos": pos.copy(),
                        "yaw": wp1["yaw"],
                    }
                )

            dense.append(wp1)

        return dense

    def _avoid_obstacles(self, waypoints, obstacles):
        """Insert obstacle detour points on unsafe waypoint segments."""
        if len(waypoints) <= 1 or len(obstacles) == 0:
            return waypoints

        output = [waypoints[0]]

        for wp0, wp1 in zip(waypoints[:-1], waypoints[1:]):
            if self._is_gate_axis_segment(wp0, wp1):
                output.append(wp1)
                continue

            detours = self._obstacle_detours(wp0, wp1, obstacles)
            for detour in detours:
                output.append(detour)
            output.append(wp1)

        return output

    @staticmethod
    def _is_gate_axis_segment(wp0, wp1):
        if int(wp0["gate_id"]) != int(wp1["gate_id"]):
            return False

        axis_pairs = {
            ("entry", "center"),
            ("center", "exit"),
        }
        return (wp0["type"], wp1["type"]) in axis_pairs

    def _prune_redundant_waypoints(self, waypoints, estimated_map=None):
        """Remove ordinary near-collinear detours only when the direct segment stays safe."""
        if len(waypoints) <= 2:
            return waypoints

        pruned = [waypoints[0]]
        removed = 0

        for i in range(1, len(waypoints) - 1):
            prev_wp = pruned[-1]
            curr_wp = waypoints[i]
            next_wp = waypoints[i + 1]

            prev_pos = np.asarray(prev_wp["pos"], dtype=float)
            curr_pos = np.asarray(curr_wp["pos"], dtype=float)
            next_pos = np.asarray(next_wp["pos"], dtype=float)

            reason = self._waypoint_prune_reason(
                prev_wp,
                curr_wp,
                next_wp,
                estimated_map,
            )
            if reason is not None:
                removed += 1
                if self.debug and self._prune_debug_prints < 20:
                    print(
                        "[WAYPOINT_PRUNE]",
                        "removed=", curr_wp["type"],
                        "gate=", curr_wp["gate_id"],
                        "reason=", reason,
                        "pos=", np.round(curr_pos, 3).tolist(),
                    )
                    self._prune_debug_prints += 1
                continue

            # Keep the previous duplicate-removal behavior, but never apply it
            # to protected gate-axis waypoints.
            if (
                np.linalg.norm(curr_pos - prev_pos) < 0.06
                and not self._is_protected_waypoint(curr_wp)
            ):
                removed += 1
                if self.debug and self._prune_debug_prints < 20:
                    print(
                        "[WAYPOINT_PRUNE]",
                        "removed=", curr_wp["type"],
                        "gate=", curr_wp["gate_id"],
                        "reason=near_duplicate_legacy",
                        "pos=", np.round(curr_pos, 3).tolist(),
                    )
                    self._prune_debug_prints += 1
                continue

            direct = float(np.linalg.norm(next_pos - prev_pos))
            via = float(np.linalg.norm(curr_pos - prev_pos) + np.linalg.norm(next_pos - curr_pos))

            detour_like = curr_wp["type"] in {
                "obstacle_detour",
                "gate_frame_detour",
                "intermediate",
            }
            same_gate = (
                int(prev_wp["gate_id"]) == int(curr_wp["gate_id"]) == int(next_wp["gate_id"])
            )

            if (
                detour_like
                and same_gate
                and direct < 0.45
                and via > 2.4 * max(direct, 1e-6)
            ):
                removed += 1
                if self.debug and self._prune_debug_prints < 20:
                    print(
                        "[WAYPOINT_PRUNE]",
                        "removed=", curr_wp["type"],
                        "gate=", curr_wp["gate_id"],
                        "reason=short_turnback_legacy",
                        "pos=", np.round(curr_pos, 3).tolist(),
                    )
                    self._prune_debug_prints += 1
                continue

            pruned.append(curr_wp)

        final_gap = np.linalg.norm(
            np.asarray(waypoints[-1]["pos"], dtype=float)
            - np.asarray(pruned[-1]["pos"], dtype=float)
        )
        if final_gap >= 0.06:
            pruned.append(waypoints[-1])

        if self.debug and removed > 0 and self._prune_debug_prints < 20:
            print("[WAYPOINT_PRUNE]", "removed_total=", removed)
            self._prune_debug_prints += 1

        return pruned

    def _waypoint_prune_reason(self, prev_wp, curr_wp, next_wp, estimated_map):
        if self._is_protected_waypoint(curr_wp):
            return None

        if curr_wp.get("type") not in {
            "intermediate",
            "obstacle_detour",
            "gate_frame_detour",
        }:
            return None

        prev_pos = np.asarray(prev_wp["pos"], dtype=float)
        curr_pos = np.asarray(curr_wp["pos"], dtype=float)
        next_pos = np.asarray(next_wp["pos"], dtype=float)

        local_shape_ok = (
            self._is_nearly_collinear(prev_pos, curr_pos, next_pos)
            or np.linalg.norm(curr_pos - prev_pos) < 0.08
            or np.linalg.norm(next_pos - curr_pos) < 0.08
        )
        if not local_shape_ok:
            return None

        if not self._segment_clear_after_pruning(prev_wp, next_wp, estimated_map):
            return None

        if self._pruning_reduces_clearance(prev_pos, curr_pos, next_pos, estimated_map):
            return None

        if self._is_nearly_collinear(prev_pos, curr_pos, next_pos):
            return "collinear"
        return "near_duplicate"

    @staticmethod
    def _is_protected_waypoint(waypoint):
        return waypoint.get("type") in {
            "entry",
            "align",
            "center",
            "exit",
            "dynamic_probe",
            "previous_gate_recross",
        }

    @staticmethod
    def _is_nearly_collinear(prev_pos, curr_pos, next_pos):
        v0 = curr_pos - prev_pos
        v1 = next_pos - curr_pos
        n0 = np.linalg.norm(v0)
        n1 = np.linalg.norm(v1)
        if n0 < 1e-6 or n1 < 1e-6:
            return True

        cos_angle = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
        angle = float(np.arccos(cos_angle))
        lateral = float(
            np.linalg.norm(np.cross(v0, next_pos - prev_pos))
            / (np.linalg.norm(next_pos - prev_pos) + 1e-9)
        )
        return angle < 0.20 and lateral < 0.08

    def _segment_clear_after_pruning(self, prev_wp, next_wp, estimated_map):
        if estimated_map is None:
            return False

        return not self._segment_between_waypoints_needs_replan(
            prev_wp,
            next_wp,
            estimated_map,
        )

    def _pruning_reduces_clearance(self, prev_pos, curr_pos, next_pos, estimated_map):
        if estimated_map is None:
            return True

        obstacles = estimated_map.get("obstacles", [])
        gates = estimated_map.get("gates", [])

        before_clearance = min(
            self._segment_obstacle_clearance(prev_pos, curr_pos, obstacles),
            self._segment_obstacle_clearance(curr_pos, next_pos, obstacles),
        )
        after_clearance = self._segment_obstacle_clearance(prev_pos, next_pos, obstacles)
        if after_clearance + 0.02 < before_clearance:
            return True

        before_frame_risk = max(
            self._route_gate_frame_risk([prev_pos, curr_pos], gates),
            self._route_gate_frame_risk([curr_pos, next_pos], gates),
        )
        after_frame_risk = self._route_gate_frame_risk([prev_pos, next_pos], gates)
        return after_frame_risk > before_frame_risk + 0.03

    def _segment_obstacle_clearance(self, p0, p1, obstacles):
        if obstacles is None or len(obstacles) == 0:
            return 0.0

        samples = [
            (1.0 - alpha) * p0 + alpha * p1
            for alpha in np.linspace(0.0, 1.0, 7)
        ]
        return self._min_obstacle_clearance(samples, obstacles)

    def _shortcut_clear_waypoints(self, waypoints, estimated_map):
        """Repeatedly remove non-critical points when direct segments stay clear."""
        if len(waypoints) <= 2:
            return waypoints

        critical_types = {"entry", "align", "center", "exit"}
        changed = True
        output = waypoints

        while changed:
            changed = False
            pruned = [output[0]]
            i = 1

            while i < len(output) - 1:
                prev_wp = pruned[-1]
                curr_wp = output[i]
                next_wp = output[i + 1]

                can_remove = (
                    curr_wp["type"] not in critical_types
                    and not self._is_protected_waypoint(curr_wp)
                )
                if can_remove and not self._segment_between_waypoints_needs_replan(
                    prev_wp,
                    next_wp,
                    estimated_map,
                ):
                    changed = True
                    i += 1
                    continue

                pruned.append(curr_wp)
                i += 1

            pruned.append(output[-1])
            output = pruned

        return output

    def _segment_between_waypoints_needs_replan(self, wp0, wp1, estimated_map):
        return self._segment_needs_replan(
            np.asarray(wp0["pos"], dtype=float),
            np.asarray(wp1["pos"], dtype=float),
            wp1,
            estimated_map,
        )

    def _push_waypoints_away_from_obstacles(self, waypoints, obstacles):
        """Move non-critical waypoints outside the inflated obstacle radius."""
        if len(waypoints) == 0 or len(obstacles) == 0:
            return waypoints

        pushed = []
        clearance_radius = self.obstacle_safety_radius + self.obstacle_detour_margin

        for wp in waypoints:
            new_wp = dict(wp)
            pos = np.asarray(wp["pos"], dtype=float).copy()

            # Keep gate-critical waypoints exact so obstacle clearance does not
            # bend the intended entry/center/exit line into the frame.
            if wp["type"] in {"entry", "align", "center", "exit"}:
                pushed.append(new_wp)
                continue

            total_push = np.zeros(2, dtype=float)
            for obstacle in obstacles:
                obstacle_pos = np.asarray(obstacle["pos_world"], dtype=float)
                offset = pos[:2] - obstacle_pos[:2]
                dist = float(np.linalg.norm(offset))

                if dist >= clearance_radius:
                    continue

                if dist < 1e-6:
                    direction = np.array([1.0, 0.0], dtype=float)
                    deficit = clearance_radius
                else:
                    direction = offset / dist
                    deficit = clearance_radius - dist

                total_push += direction * deficit

            pos[:2] += total_push
            new_wp["pos"] = pos
            pushed.append(new_wp)

        return pushed

    def _obstacle_detours(self, wp0, wp1, obstacles):
        """Return lateral detours for obstacles intersecting a segment."""
        p0 = np.asarray(wp0["pos"], dtype=float)
        p1 = np.asarray(wp1["pos"], dtype=float)
        segment_xy = p1[:2] - p0[:2]
        seg_len_sq = float(np.dot(segment_xy, segment_xy))
        if seg_len_sq < 1e-8:
            return []

        clearance_radius = self.obstacle_safety_radius + self.obstacle_detour_margin
        hits = []

        for obstacle in obstacles:
            obstacle_pos = np.asarray(obstacle["pos_world"], dtype=float)
            alpha = float(
                np.clip(
                    np.dot(obstacle_pos[:2] - p0[:2], segment_xy) / seg_len_sq,
                    0.0,
                    1.0,
                )
            )
            closest_xy = p0[:2] + alpha * segment_xy
            offset_xy = closest_xy - obstacle_pos[:2]
            dist_xy = float(np.linalg.norm(offset_xy))

            if dist_xy >= clearance_radius:
                continue

            hits.append((alpha, obstacle_pos, clearance_radius - dist_xy))

        if not hits:
            return []

        hits.sort(key=lambda item: item[0])
        side = self._choose_obstacle_side(p0, p1, segment_xy, hits, obstacles)
        detours = []

        for alpha, obstacle_pos, _deficit in hits:
            candidate = self._obstacle_detour_candidate(p0, p1, obstacle_pos, alpha, side)
            detours.append(
                {
                    "type": "obstacle_detour",
                    "gate_id": int(wp1["gate_id"]),
                    "pos": candidate,
                    "yaw": wp1["yaw"],
                }
            )

        return detours

    def _choose_obstacle_side(self, p0, p1, segment_xy, hits, obstacles):
        side = np.array([-segment_xy[1], segment_xy[0]], dtype=float)
        side_norm = np.linalg.norm(side)
        if side_norm < 1e-6:
            side = np.array([1.0, 0.0], dtype=float)
        else:
            side = side / side_norm

        candidates = []
        for sign in (1.0, -1.0):
            points = [
                self._obstacle_detour_candidate(p0, p1, obstacle_pos, alpha, sign * side)
                for alpha, obstacle_pos, _deficit in hits
            ]
            score = self._score_detour_route(
                p0,
                p1,
                points,
                obstacles=obstacles,
                gates=None,
                label="obstacle",
            )
            candidates.append((score, sign * side))

        return max(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _obstacle_escape_direction(segment_xy, offset_xy, dist_xy):
        if dist_xy > 1e-6:
            return offset_xy / dist_xy

        side = np.array([-segment_xy[1], segment_xy[0]], dtype=float)
        side_norm = np.linalg.norm(side)
        if side_norm < 1e-6:
            return np.array([1.0, 0.0], dtype=float)
        return side / side_norm

    def _obstacle_detour_candidate(self, p0, p1, obstacle_pos, alpha, direction_xy):
        z = (1.0 - alpha) * p0[2] + alpha * p1[2]
        clearance_radius = self.obstacle_safety_radius + self.obstacle_detour_margin
        return np.array(
            [
                obstacle_pos[0] + direction_xy[0] * clearance_radius,
                obstacle_pos[1] + direction_xy[1] * clearance_radius,
                z,
            ],
            dtype=float,
        )

    @staticmethod
    def _choose_safer_obstacle_candidate(p0, p1, candidate_a, candidate_b, obstacles):
        obstacle_positions = np.asarray(
            [obstacle["pos_world"] for obstacle in obstacles],
            dtype=float,
        )
        dists_a = np.linalg.norm(candidate_a[:2] - obstacle_positions[:, :2], axis=1)
        dists_b = np.linalg.norm(candidate_b[:2] - obstacle_positions[:, :2], axis=1)

        min_a = float(np.min(dists_a))
        min_b = float(np.min(dists_b))
        via_a = float(np.linalg.norm(candidate_a - p0) + np.linalg.norm(p1 - candidate_a))
        via_b = float(np.linalg.norm(candidate_b - p0) + np.linalg.norm(p1 - candidate_b))

        # Prefer the shorter side when both candidates are similarly safe.
        safety_delta = min_a - min_b
        if abs(safety_delta) < 0.08:
            return candidate_a if via_a <= via_b else candidate_b

        score_a = min_a - 0.12 * via_a
        score_b = min_b - 0.12 * via_b
        return candidate_a if score_a >= score_b else candidate_b

    def _avoid_gate_frames(self, waypoints, gates):
        """Insert detours when a segment would cross a gate frame."""
        if len(waypoints) <= 1 or len(gates) == 0:
            return waypoints

        output = [waypoints[0]]

        for wp0, wp1 in zip(waypoints[:-1], waypoints[1:]):
            detour = self._gate_frame_detour(wp0, wp1, gates)
            if detour is not None:
                output.append(detour)
            output.append(wp1)

        return output

    def _gate_frame_detour(self, wp0, wp1, gates):
        """Choose the best single detour around nearby gate frame geometry."""
        p0 = np.asarray(wp0["pos"], dtype=float)
        p1 = np.asarray(wp1["pos"], dtype=float)
        segment = p1 - p0
        segment_norm = np.linalg.norm(segment)
        if segment_norm < 1e-6:
            return None

        segment_dir = segment / segment_norm
        best_detour = None
        best_score = -np.inf

        for gate in gates:
            gate_id = int(gate["id"])
            gate_pos = np.asarray(gate["pos_world"], dtype=float)
            normal = np.asarray(gate.get("normal", self._normal_from_yaw(gate["yaw"])), dtype=float)
            normal = normal / (np.linalg.norm(normal) + 1e-9)

            y_axis = np.array([-normal[1], normal[0], 0.0], dtype=float)
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
            z_axis = np.array([0.0, 0.0, 1.0], dtype=float)

            # The intended entry/center/exit segments for this gate are allowed
            # to pass through the central opening, but not through the frame ring.
            aligned_with_gate = abs(float(np.dot(segment_dir, normal))) > 0.78
            same_gate_segment = int(wp0["gate_id"]) == gate_id and int(wp1["gate_id"]) == gate_id

            detours = self._segment_gate_frame_detours(
                p0,
                p1,
                segment_dir,
                gate_pos,
                normal,
                y_axis,
                z_axis,
                allow_center_opening=same_gate_segment and aligned_with_gate,
            )
            if not detours:
                continue

            for detour in detours:
                score = self._score_detour_route(
                    p0,
                    p1,
                    [detour["pos"]],
                    obstacles=None,
                    gates=gates,
                    label="gate_frame",
                )
                if score > best_score:
                    best_score = score
                    best_detour = {
                        "type": "gate_frame_detour",
                        "gate_id": int(wp1["gate_id"]),
                        "pos": detour["pos"],
                        "yaw": wp1["yaw"],
                    }

        return best_detour

    def _segment_gate_frame_detours(
        self,
        p0,
        p1,
        segment_dir,
        gate_pos,
        normal,
        y_axis,
        z_axis,
        allow_center_opening,
    ):
        """Generate side/top candidate detours for one gate-frame violation."""
        alphas = np.linspace(0.0, 1.0, 17)
        samples = p0[None, :] + alphas[:, None] * (p1 - p0)[None, :]
        rel = samples - gate_pos

        x_local = rel @ normal
        y_local = rel @ y_axis
        z_local = rel @ z_axis

        near_plane = np.abs(x_local) <= self.gate_frame_depth
        inside_outer_box = (
            (np.abs(y_local) <= self.gate_outer_half_width + self.gate_frame_clearance)
            & (np.abs(z_local) <= self.gate_outer_half_height + self.gate_frame_clearance)
        )
        inside_safe_opening = (
            (np.abs(y_local) <= self.gate_opening_half_width - self.gate_frame_clearance)
            & (np.abs(z_local) <= self.gate_opening_half_height - self.gate_frame_clearance)
        )

        if allow_center_opening:
            violation = near_plane & inside_outer_box & ~inside_safe_opening
        else:
            violation = near_plane & inside_outer_box

        if not np.any(violation):
            return None

        violation_indices = np.where(violation)[0]
        center_score = np.abs(y_local[violation_indices]) + 0.6 * np.abs(z_local[violation_indices])
        idx = int(violation_indices[np.argmin(center_score)])
        closest = samples[idx]
        y_hit = float(y_local[idx])
        z_hit = float(z_local[idx])
        x_hit = float(x_local[idx])

        side_sign = np.sign(y_hit) if abs(y_hit) > 1e-6 else np.sign(np.dot(segment_dir, y_axis))
        if abs(side_sign) < 1e-6:
            side_sign = 1.0

        side_target = side_sign * (self.gate_outer_half_width + self.gate_frame_clearance)
        top_target = self.gate_outer_half_height + self.gate_frame_clearance

        side_move = abs(side_target - y_hit)
        top_move = abs(top_target - z_hit)

        plane_sign = np.sign(x_hit) if abs(x_hit) > 1e-6 else -1.0
        plane_offset = plane_sign * 0.14 * normal

        candidate_offsets = [
            (side_target - y_hit) * y_axis,
            (-side_target - y_hit) * y_axis,
            (top_target - z_hit) * z_axis,
        ]
        if side_move > 1.35 * top_move:
            candidate_offsets = [
                (top_target - z_hit) * z_axis,
                (side_target - y_hit) * y_axis,
                (-side_target - y_hit) * y_axis,
            ]

        detours = []
        for offset in candidate_offsets:
            detour = closest + offset + plane_offset
            if not any(np.linalg.norm(detour - item["pos"]) < 1e-6 for item in detours):
                detours.append({"pos": detour})

        return detours

    def _score_detour_route(
        self,
        start_pos,
        final_pos,
        detour_points,
        obstacles=None,
        gates=None,
        label=None,
    ):
        """Rank detours by clearance, path length, turn smoothness, and frame risk."""
        route = (
            [np.asarray(start_pos, dtype=float)]
            + [np.asarray(point, dtype=float) for point in detour_points]
            + [np.asarray(final_pos, dtype=float)]
        )
        route_length = self._route_length(route)
        direct_length = float(np.linalg.norm(route[-1] - route[0]))
        extra_distance = max(0.0, route_length - direct_length)

        # Positive obstacle clearance means the candidate stays outside the
        # inflated pole radius. Gate frame risk is zero in the gate opening
        # center corridor, so old-gate recross through the opening remains legal.
        min_clearance = self._min_obstacle_clearance(route[1:-1], obstacles)
        turn_penalty = self._turn_angle_penalty(route)
        gate_frame_risk = self._route_gate_frame_risk(route, gates)

        score = (
            1.25 * min_clearance
            - 0.85 * extra_distance
            - 0.22 * turn_penalty
            - 0.75 * gate_frame_risk
        )

        if self.debug and label is not None and self._clearance_debug_prints < 12:
            print(
                "[CLEARANCE_SCORE]",
                "label=", label,
                "score=", round(score, 3),
                "clearance=", round(min_clearance, 3),
                "extra=", round(extra_distance, 3),
                "turn=", round(turn_penalty, 3),
                "frame_risk=", round(gate_frame_risk, 3),
            )
            self._clearance_debug_prints += 1

        return score

    @staticmethod
    def _route_length(route):
        return sum(
            float(np.linalg.norm(route[i + 1] - route[i]))
            for i in range(len(route) - 1)
        )

    def _min_obstacle_clearance(self, points, obstacles):
        if not points or obstacles is None or len(obstacles) == 0:
            return 0.0

        obstacle_positions = np.asarray(
            [obstacle["pos_world"] for obstacle in obstacles],
            dtype=float,
        )
        clearance_radius = self.obstacle_safety_radius + self.obstacle_detour_margin
        min_clearance = np.inf

        for point in points:
            point = np.asarray(point, dtype=float)
            dists = np.linalg.norm(point[:2] - obstacle_positions[:, :2], axis=1)
            min_clearance = min(min_clearance, float(np.min(dists)) - clearance_radius)

        if not np.isfinite(min_clearance):
            return 0.0
        return min_clearance

    @staticmethod
    def _turn_angle_penalty(route):
        if len(route) < 3:
            return 0.0

        penalty = 0.0
        for i in range(1, len(route) - 1):
            v0 = route[i] - route[i - 1]
            v1 = route[i + 1] - route[i]
            n0 = np.linalg.norm(v0)
            n1 = np.linalg.norm(v1)
            if n0 < 1e-6 or n1 < 1e-6:
                continue
            cos_angle = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
            penalty += float(np.arccos(cos_angle) / np.pi)

        return penalty

    def _route_gate_frame_risk(self, route, gates):
        """Estimate the maximum gate-frame risk along a polyline route."""
        if gates is None or len(gates) == 0 or len(route) < 2:
            return 0.0

        max_risk = 0.0
        for p0, p1 in zip(route[:-1], route[1:]):
            for alpha in np.linspace(0.0, 1.0, 5):
                point = (1.0 - alpha) * p0 + alpha * p1
                for gate in gates:
                    max_risk = max(max_risk, self._gate_frame_point_risk(point, gate))

        return max_risk

    def _gate_frame_point_risk(self, point, gate):
        """Score how close a point is to gate frame material."""
        gate_pos, normal, lateral_axis = self._gate_position_normal_lateral(gate)
        rel = np.asarray(point, dtype=float) - gate_pos
        x_local = float(np.dot(rel, normal))
        y_local = float(np.dot(rel, lateral_axis))
        z_local = float(rel[2])

        plane_extent = self.gate_frame_depth + 0.25
        plane_factor = max(0.0, 1.0 - abs(x_local) / plane_extent)
        if plane_factor <= 0.0:
            return 0.0

        outer_y = self.gate_outer_half_width + self.gate_frame_clearance
        outer_z = self.gate_outer_half_height + self.gate_frame_clearance
        inside_outer = abs(y_local) <= outer_y and abs(z_local) <= outer_z
        if not inside_outer:
            return 0.0

        opening_y = max(0.0, self.gate_opening_half_width - 0.04)
        opening_z = max(0.0, self.gate_opening_half_height - 0.04)
        inside_opening_center = abs(y_local) <= opening_y and abs(z_local) <= opening_z
        if inside_opening_center:
            return 0.0

        frame_depth = min(outer_y - abs(y_local), outer_z - abs(z_local))
        frame_depth = max(0.0, frame_depth)
        return plane_factor * (1.0 + frame_depth)

    @staticmethod
    def _normal_from_yaw(yaw):
        normal = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
        return normal / (np.linalg.norm(normal) + 1e-9)

    def _compute_reference_velocity(
        self,
        drone_pos,
        target_pos,
        waypoint=None,
        estimated_map=None,
        target_gate=None,
    ):
        """Apply risk-aware speed scheduling at the reference-velocity layer."""
        error = target_pos - drone_pos
        distance = np.linalg.norm(error)

        if distance < 1e-6:
            return np.zeros(3)

        direction = error / distance
        speed_scale = 0.65 if self._update_tick <= self._slowdown_until_tick else 1.0
        
        if waypoint is not None:
            risk_scale = self._risk_aware_speed_scale(
                drone_pos,
                target_pos,
                waypoint,
                estimated_map,
                target_gate,
            )
            speed_scale = min(speed_scale, risk_scale)

        speed_scale = self._smooth_speed_scale(speed_scale)

        speed = min(self.max_speed * speed_scale, distance)

        return direction * speed

    def _risk_aware_speed_scale(
        self,
        drone_pos,
        target_pos,
        waypoint,
        estimated_map,
        target_gate,
    ):
        """Combine waypoint type, clearance, frame risk, and turning risk."""
        wp_type = waypoint.get("type")
        scale = self._speed_scale_for_waypoint_type(waypoint, target_gate)
        reasons = [wp_type]

        if estimated_map is not None:
            clearance_scale, clearance = self._clearance_speed_scale(target_pos, estimated_map)
            frame_scale, frame_risk = self._gate_frame_speed_scale(
                drone_pos,
                target_pos,
                estimated_map,
            )
            turn_scale, turn = self._curvature_speed_scale(drone_pos, target_pos)
            scale = min(scale, clearance_scale, frame_scale, turn_scale)

            if clearance_scale < 1.0:
                reasons.append("clearance")
            if frame_scale < 1.0:
                reasons.append("frame")
            if turn_scale < 1.0:
                reasons.append("turn")
        else:
            clearance = 0.0
            frame_risk = 0.0
            turn = 0.0

        if self._prev_gate_recross_active:
            scale = min(scale, self.gate_pass_speed_scale)
            reasons.append("recross")

        upper_scale = 1.16 if self.aggressive_speed else 1.08
        scale = float(np.clip(scale, 0.45, upper_scale))

        if self.debug and self._update_tick % 10 == 0 and self._risk_speed_debug_prints < 16:
            print(
                "[RISK_SPEED]",
                "wp_id=", self.current_wp_id,
                "type=", wp_type,
                "scale=", round(scale, 3),
                "clearance=", round(clearance, 3),
                "turn=", round(turn, 3),
                "frame_risk=", round(frame_risk, 3),
                "reason=", ",".join(reasons),
            )
            self._risk_speed_debug_prints += 1

        return scale

    def _speed_scale_for_waypoint_type(self, waypoint, target_gate):
        """Return the baseline speed scale for each waypoint type."""
        wp_type = waypoint.get("type")

        if self.aggressive_speed:
            if wp_type in {"align", "center", "previous_gate_recross", "dynamic_probe"}:
                return min(max(self.gate_pass_speed_scale, 0.68), 0.82)
            if wp_type == "entry":
                return 0.92
            if wp_type == "exit":
                if target_gate is not None and int(waypoint.get("gate_id", -1)) != int(target_gate):
                    return 0.76
                return 0.96
            if wp_type == "obstacle_detour":
                return 0.90
            if wp_type == "gate_frame_detour":
                return 0.84
            if wp_type == "dynamic_replan":
                return 1.08
            if wp_type == "intermediate":
                return 1.12

        if wp_type in {"align", "center", "previous_gate_recross", "dynamic_probe"}:
            return min(self.gate_pass_speed_scale, 0.58)

        if wp_type == "entry":
            return 0.74

        if wp_type == "exit":
            if target_gate is not None and int(waypoint.get("gate_id", -1)) != int(target_gate):
                return 0.62
            return 0.82

        if wp_type == "obstacle_detour":
            return 0.78

        if wp_type == "gate_frame_detour":
            return 0.72

        if wp_type == "dynamic_replan":
            return 1.03

        if wp_type == "intermediate":
            return 1.06

        return 1.0

    def _clearance_speed_scale(self, target_pos, estimated_map):
        obstacles = estimated_map.get("obstacles", [])
        clearance = self._min_obstacle_clearance([target_pos], obstacles)

        if self.aggressive_speed:
            if not obstacles:
                return 1.0, clearance
            if clearance < 0.02:
                return 0.66, clearance
            if clearance < 0.10:
                return 0.84, clearance
            if clearance < 0.18:
                return 1.00, clearance
            return 1.12, clearance

        if not obstacles:
            return 1.0, clearance
        if clearance < 0.02:
            return 0.62, clearance
        if clearance < 0.10:
            return 0.78, clearance
        if clearance < 0.18:
            return 0.92, clearance
        return 1.08, clearance

    def _gate_frame_speed_scale(self, drone_pos, target_pos, estimated_map):
        gates = estimated_map.get("gates", [])
        risk = self._route_gate_frame_risk([drone_pos, target_pos], gates)

        if self.aggressive_speed:
            if risk > 0.75:
                return 0.62, risk
            if risk > 0.35:
                return 0.80, risk
            if risk > 0.12:
                return 0.98, risk
            return 1.12, risk

        if risk > 0.75:
            return 0.58, risk
        if risk > 0.35:
            return 0.72, risk
        if risk > 0.12:
            return 0.88, risk
        return 1.08, risk

    def _curvature_speed_scale(self, drone_pos, target_pos):
        if self.current_wp_id + 1 >= len(self.waypoints):
            return 1.0, 0.0

        next_pos = np.asarray(self.waypoints[self.current_wp_id + 1]["pos"], dtype=float)
        route = [
            np.asarray(drone_pos, dtype=float),
            np.asarray(target_pos, dtype=float),
            next_pos,
        ]
        turn = self._turn_angle_penalty(route)

        if self.aggressive_speed:
            if turn > 0.55:
                return 0.78, turn
            if turn > 0.34:
                return 0.92, turn
            if turn > 0.18:
                return 1.02, turn
            return 1.12, turn

        if turn > 0.55:
            return 0.70, turn
        if turn > 0.34:
            return 0.84, turn
        if turn > 0.18:
            return 0.96, turn
        return 1.06, turn

    def _smooth_speed_scale(self, speed_scale):
        """Ramp speed increases while applying slowdowns quickly."""
        # Slowdowns should apply immediately; speedups are ramped to avoid
        # abrupt attitude commands on short segments.
        alpha = 0.45 if speed_scale < self._last_speed_scale else 0.18
        smoothed = (1.0 - alpha) * self._last_speed_scale + alpha * speed_scale
        upper_scale = 1.16 if self.aggressive_speed else 1.08
        self._last_speed_scale = float(np.clip(smoothed, 0.45, upper_scale))
        return self._last_speed_scale

    def _hover_reference(self, self_state):
        return {
            "pos": self_state["pos_world"].copy(),
            "vel": np.zeros(3),
            "acc": np.zeros(3),
            "yaw": self_state["yaw_world"],
            "omega": np.zeros(3),
            "wp_id": None,
            "wp_type": "hover",
            "gate_id": None,
        }

    def print_waypoints(self, estimated_map):
        print("\n========== WAYPOINT DEBUG ==========")
        print("num gates:", len(estimated_map["gates"]))
        print("num waypoints:", len(self.waypoints))
        print("current_wp_id:", self.current_wp_id)

        for i, wp in enumerate(self.waypoints):
            print(
                f"{i:02d} | gate={wp['gate_id']} | "
                f"type={wp['type']} | pos={np.round(wp['pos'], 2)}"
            )

        print("====================================\n")
