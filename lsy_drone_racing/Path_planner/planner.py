"""Simple sequential gate planner kept as a readable baseline.

The main controller now uses TrajectoryManager for obstacle-aware planning.
DirectionalGatePlanner remains useful for debugging the basic
entry-center-exit gate logic.
"""

import numpy as np


class DirectionalGatePlanner:
    """Directional sequential gate planner.

    For each gate:
        APPROACH -> PASS -> EXIT -> next gate
    """

    APPROACH = "approach"
    PASS = "pass"
    EXIT = "exit"

    def __init__(
        self,
        entry_distance=0.42,
        exit_distance=0.50,
        entry_threshold=0.18,
        center_threshold=0.22,
        exit_threshold=0.25,
    ):
        """Initialize phase thresholds and gate offsets."""
        self.current_gate_id = 0
        self.phase = self.APPROACH

        self.entry_distance = entry_distance
        self.exit_distance = exit_distance

        self.entry_threshold = entry_threshold
        self.center_threshold = center_threshold
        self.exit_threshold = exit_threshold

    def update(self, estimated_map, self_state):
        """Advance the phase machine and return the next target pose."""
        gates = estimated_map["gates"]

        if len(gates) == 0:
            return self._hover_target(self_state)

        if self.current_gate_id >= len(gates):
            self.current_gate_id = 0

        gate = gates[self.current_gate_id]
        drone_pos = self_state["pos_world"]

        gate_pos = gate["pos_world"]
        yaw = gate["yaw"]

        normal = np.array([
            np.cos(yaw),
            np.sin(yaw),
            0.0,
        ])

        entry_pos = gate_pos - normal * self.entry_distance
        center_pos = gate_pos.copy()
        exit_pos = gate_pos + normal * self.exit_distance

        # The phase machine forces a clean approach, gate-center pass, and
        # exit before moving to the next gate.
        if self.phase == self.APPROACH:
            target_pos = entry_pos
            threshold = self.entry_threshold

            if np.linalg.norm(drone_pos - entry_pos) < threshold:
                self.phase = self.PASS

        elif self.phase == self.PASS:
            target_pos = center_pos
            threshold = self.center_threshold

            if np.linalg.norm(drone_pos - center_pos) < threshold:
                self.phase = self.EXIT

        else:
            target_pos = exit_pos
            threshold = self.exit_threshold

            if np.linalg.norm(drone_pos - exit_pos) < threshold:
                self.current_gate_id += 1
                self.phase = self.APPROACH

                if self.current_gate_id >= len(gates):
                    self.current_gate_id = 0

        pos_local = (
            self_state["world_to_local"](target_pos)
            if "world_to_local" in self_state
            else target_pos - drone_pos
        )

        return {
            "gate_id": self.current_gate_id,
            "phase": self.phase,
            "pos_world": target_pos.copy(),
            "pos_local": pos_local.copy(),
            "yaw": yaw,
            "source": gate["source"],
        }

    def _hover_target(self, self_state):
        """Return a hold-position target when no gates are available."""
        return {
            "gate_id": None,
            "phase": "hover",
            "pos_world": self_state["pos_world"].copy(),
            "pos_local": self_state["pos_local"].copy(),
            "yaw": self_state["yaw_world"],
            "source": "hover",
        }

    def print_summary(self, target):
        """Print the current baseline-planner target."""
        print("\n========== PLANNER DEBUG ==========")
        print("Current gate:", target["gate_id"])
        print("Phase       :", target["phase"])
        print("Target world:", target["pos_world"])
        print("Target local:", target["pos_local"])
        print("Target yaw  :", target["yaw"])
        print("Source      :", target["source"])
        print("===================================\n")
