import numpy as np


class TrajectoryManager:
    def __init__(
        self,
        waypoint_threshold=0.20,
        entry_distance=0.42,
        exit_distance=0.58,
        max_speed=0.25,
        max_segment_length=0.45,
        obstacle_safety_radius=0.36,
        obstacle_detour_margin=0.08,
        enable_dynamic_replan=True,
        loop=True,
        debug=False,
    ):
        self.waypoint_threshold = waypoint_threshold
        self.entry_distance = entry_distance
        self.exit_distance = exit_distance
        self.max_speed = max_speed
        self.max_segment_length = max_segment_length
        self.enable_dynamic_replan = enable_dynamic_replan
        self.loop = loop
        self.debug = debug

        self.waypoints = []
        self.current_wp_id = 0
        self.initialized = False
        self._last_target_gate = None
        self._last_dynamic_replan_tick = -100
        self._update_tick = 0
        self._slowdown_until_tick = -1

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

        waypoints = self._avoid_obstacles(waypoints, estimated_map["obstacles"])
        waypoints = self._avoid_gate_frames(waypoints, estimated_map["gates"])
        waypoints = self._push_waypoints_away_from_obstacles(waypoints, estimated_map["obstacles"])
        waypoints = self._avoid_obstacles(waypoints, estimated_map["obstacles"])
        waypoints = self._shortcut_clear_waypoints(waypoints, estimated_map)
        waypoints = self._prune_redundant_waypoints(waypoints)
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
        target_pos = current_wp["pos"]
        distance = np.linalg.norm(target_pos - drone_pos)

        ref_vel = self._compute_reference_velocity(drone_pos, target_pos)

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
        self.build_waypoints_from_map(
            estimated_map,
            keep_progress=True,
        )

    def _prepend_current_segment_detours(self, self_state, estimated_map):
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
        local = self._prune_redundant_waypoints(local)
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

    def _threshold_for_waypoint(self, waypoint):
        wp_type = waypoint["type"]

        if wp_type in {"obstacle_detour", "gate_frame_detour", "dynamic_replan"}:
            return max(self.waypoint_threshold, 0.40)

        if wp_type == "intermediate":
            return max(self.waypoint_threshold, 0.34)

        if wp_type == "center":
            return min(self.waypoint_threshold, 0.18)

        return self.waypoint_threshold

    def _segment_needs_replan(self, p0, p1, wp1, estimated_map):
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
                local = self._prune_redundant_waypoints(local)

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

                route_positions = [pos] + [np.asarray(wp["pos"], dtype=float) for wp in repaired] + [target]
                route_length = sum(
                    float(np.linalg.norm(route_positions[i + 1] - route_positions[i]))
                    for i in range(len(route_positions) - 1)
                )
                first_turn = np.linalg.norm(route_positions[1] - (pos + target_dir * forward_step))
                candidates.append((route_length + 0.35 * first_turn, repaired))

        if not candidates:
            print(
                "[TrajectoryManager] no_clear_candidate: dynamic replan tried"
                f" {len(list((0.15, 0.25, 0.35, 0.50, 0.70, 0.90))) * len((0.0, 0.18, -0.18, 0.32, -0.32, 0.48, -0.48, 0.64, -0.64, 0.80, -0.80, 1.00, -1.00)))} candidates"
            )
            return []

        return min(candidates, key=lambda item: item[0])[1]

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
        self.current_wp_id += 1

        if self.current_wp_id >= len(self.waypoints):
            if self.loop:
                self.current_wp_id = 0
            else:
                self.current_wp_id = len(self.waypoints) - 1

    def _densify_waypoints(self, waypoints):
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

    def _prune_redundant_waypoints(self, waypoints):
        if len(waypoints) <= 2:
            return waypoints

        pruned = [waypoints[0]]

        for i in range(1, len(waypoints) - 1):
            prev_wp = pruned[-1]
            curr_wp = waypoints[i]
            next_wp = waypoints[i + 1]

            prev_pos = np.asarray(prev_wp["pos"], dtype=float)
            curr_pos = np.asarray(curr_wp["pos"], dtype=float)
            next_pos = np.asarray(next_wp["pos"], dtype=float)

            if np.linalg.norm(curr_pos - prev_pos) < 0.06:
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

            if detour_like and same_gate and direct < 0.45 and via > 2.4 * max(direct, 1e-6):
                continue

            pruned.append(curr_wp)

        if np.linalg.norm(np.asarray(waypoints[-1]["pos"], dtype=float) - np.asarray(pruned[-1]["pos"], dtype=float)) >= 0.06:
            pruned.append(waypoints[-1])

        return pruned

    def _shortcut_clear_waypoints(self, waypoints, estimated_map):
        if len(waypoints) <= 2:
            return waypoints

        critical_types = {"entry", "center", "exit"}
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

                can_remove = curr_wp["type"] not in critical_types
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
        if len(waypoints) == 0 or len(obstacles) == 0:
            return waypoints

        pushed = []
        clearance_radius = self.obstacle_safety_radius + self.obstacle_detour_margin

        for wp in waypoints:
            new_wp = dict(wp)
            pos = np.asarray(wp["pos"], dtype=float).copy()

            # Keep gate-critical waypoints exact so obstacle clearance does not
            # bend the intended entry/center/exit line into the frame.
            if wp["type"] in {"entry", "center", "exit"}:
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
            route = [p0] + points + [p1]
            length = sum(
                float(np.linalg.norm(route[i + 1] - route[i]))
                for i in range(len(route) - 1)
            )
            obstacle_positions = np.asarray([obstacle["pos_world"] for obstacle in obstacles], dtype=float)
            min_clearance = min(
                float(np.min(np.linalg.norm(point[:2] - obstacle_positions[:, :2], axis=1)))
                for point in points
            )
            candidates.append((length - 0.25 * min_clearance, sign * side))

        return min(candidates, key=lambda item: item[0])[1]

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
        obstacle_positions = np.asarray([obstacle["pos_world"] for obstacle in obstacles], dtype=float)
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

            detour = self._segment_gate_frame_detour(
                p0,
                p1,
                segment_dir,
                gate_pos,
                normal,
                y_axis,
                z_axis,
                allow_center_opening=same_gate_segment and aligned_with_gate,
            )
            if detour is None:
                continue

            midpoint = 0.5 * (p0 + p1)
            score = np.linalg.norm(detour["pos"] - midpoint)
            if score > best_score:
                best_score = score
                best_detour = {
                    "type": "gate_frame_detour",
                    "gate_id": int(wp1["gate_id"]),
                    "pos": detour["pos"],
                    "yaw": wp1["yaw"],
                }

        return best_detour

    def _segment_gate_frame_detour(
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

        # Prefer horizontal side detours; going over the top is a fallback when
        # the side escape is much longer.
        if side_move <= 1.35 * top_move:
            detour = closest + (side_target - y_hit) * y_axis
        else:
            detour = closest + (top_target - z_hit) * z_axis

        plane_sign = np.sign(x_hit) if abs(x_hit) > 1e-6 else -1.0
        detour = detour + plane_sign * 0.14 * normal
        return {"pos": detour}

    @staticmethod
    def _normal_from_yaw(yaw):
        normal = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
        return normal / (np.linalg.norm(normal) + 1e-9)

    def _compute_reference_velocity(self, drone_pos, target_pos):
        error = target_pos - drone_pos
        distance = np.linalg.norm(error)

        if distance < 1e-6:
            return np.zeros(3)

        direction = error / distance
        speed_scale = 0.65 if self._update_tick <= self._slowdown_until_tick else 1.0
        speed = min(self.max_speed * speed_scale, distance)

        return direction * speed

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
