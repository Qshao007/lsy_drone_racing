"""Maintain the planner's current estimate of gates and obstacles.

The map starts from nominal config positions. When an object becomes visible,
its observed pose replaces the nominal pose and is marked as seen.
"""

import numpy as np


class LocalMapping:
    """Maintain estimated gate and obstacle map."""

    def __init__(self, config_manager):
        """Initialize nominal gate and obstacle maps from ConfigManager."""
        self.config = config_manager

        self.gate_map = []
        self.obstacle_map = []

        for i, gate in enumerate(self.config.get_nominal_gates()):
            yaw = float(gate["rpy"][2])
            normal = self._normal_from_yaw(yaw)

            # Nominal objects are useful before they are visible, but the
            # planner can still tell whether each pose has been confirmed.
            self.gate_map.append({
                "id": i,
                "pos_world": gate["pos"].copy(),
                "rpy": gate["rpy"].copy(),
                "yaw": yaw,
                "normal": normal,
                "source": "nominal",
                "seen": False,
            })

        for i, obstacle in enumerate(self.config.get_nominal_obstacles()):
            self.obstacle_map.append({
                "id": i,
                "pos_world": obstacle["pos"].copy(),
                "source": "nominal",
                "seen": False,
            })

    def update(self, parsed_obs, self_location):
        """Merge visible observations into the persistent map estimate."""
        for gate in parsed_obs["gates"]:
            gate_id = gate["id"]

            if gate_id >= len(self.gate_map):
                continue

            if gate["visible"]:
                yaw = float(gate["yaw"])
                normal = gate.get("normal", self._normal_from_yaw(yaw))
                normal = normal / (np.linalg.norm(normal) + 1e-9)

                self.gate_map[gate_id]["pos_world"] = gate["pos"].copy()
                self.gate_map[gate_id]["rpy"] = gate["rpy"].copy()
                self.gate_map[gate_id]["yaw"] = yaw
                self.gate_map[gate_id]["normal"] = normal.copy()
                self.gate_map[gate_id]["source"] = gate.get("source", "observed")
                self.gate_map[gate_id]["seen"] = True

        for obstacle in parsed_obs["obstacles"]:
            obstacle_id = obstacle["id"]

            if obstacle_id >= len(self.obstacle_map):
                continue

            if obstacle["visible"]:
                self.obstacle_map[obstacle_id]["pos_world"] = obstacle["pos"].copy()
                self.obstacle_map[obstacle_id]["source"] = obstacle.get("source", "observed")
                self.obstacle_map[obstacle_id]["seen"] = True

        return self.get_map(self_location)

    def get_map(self, self_location):
        """Return estimated map in world, local, and body frame."""
        gates = []

        for gate in self.gate_map:
            pos_world = gate["pos_world"].copy()
            pos_local = self_location.world_to_local(pos_world)

            if hasattr(self_location, "world_to_body"):
                pos_body = self_location.world_to_body(pos_world)
            else:
                pos_body = pos_world - self_location.pos_world

            gates.append({
                "id": gate["id"],
                "pos_world": pos_world,
                "pos_local": pos_local,
                "pos_body": pos_body,
                "rpy": gate["rpy"].copy(),
                "yaw": gate["yaw"],
                "normal": gate["normal"].copy(),
                "source": gate["source"],
                "seen": gate["seen"],
            })

        obstacles = []

        for obstacle in self.obstacle_map:
            pos_world = obstacle["pos_world"].copy()
            pos_local = self_location.world_to_local(pos_world)

            if hasattr(self_location, "world_to_body"):
                pos_body = self_location.world_to_body(pos_world)
            else:
                pos_body = pos_world - self_location.pos_world

            obstacles.append({
                "id": obstacle["id"],
                "pos_world": pos_world,
                "pos_local": pos_local,
                "pos_body": pos_body,
                "source": obstacle["source"],
                "seen": obstacle["seen"],
            })

        return {
            "gates": gates,
            "obstacles": obstacles,
        }

    def print_summary(self, estimated_map):
        """Print a compact debug view of the current map estimate."""
        print("\n========== LOCAL MAPPING DEBUG ==========")

        print("\nGates:")
        for gate in estimated_map["gates"]:
            print(
                f"Gate {gate['id']}: "
                f"world={gate['pos_world']}, "
                f"body={gate['pos_body']}, "
                f"yaw={gate['yaw']:.2f}, "
                f"normal={gate['normal']}, "
                f"source={gate['source']}, "
                f"seen={gate['seen']}"
            )

        print("\nObstacles:")
        for obstacle in estimated_map["obstacles"]:
            print(
                f"Obstacle {obstacle['id']}: "
                f"world={obstacle['pos_world']}, "
                f"body={obstacle['pos_body']}, "
                f"source={obstacle['source']}, "
                f"seen={obstacle['seen']}"
            )

        print("========================================\n")

    @staticmethod
    def _normal_from_yaw(yaw):
        """Convert yaw into the gate's forward normal vector."""
        normal = np.array(
            [
                np.cos(yaw),
                np.sin(yaw),
                0.0,
            ],
            dtype=float,
        )
        return normal / (np.linalg.norm(normal) + 1e-9)
