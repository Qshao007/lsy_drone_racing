import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.Path_planner.config_manager import ConfigManager


class RawObservation:
    def __init__(self, config_path: str):
        self.config = ConfigManager(config_path)
        self.sensor_range = self.config.get_sensor_range()
        self.nominal_gates = self.config.get_nominal_gates()
        self.nominal_obstacles = self.config.get_nominal_obstacles()
        self.safety_limits = self.config.get_safety_limits()

    def update(self, obs, info=None):
        drone = self._read_drone_state(obs)
        gates = self._read_gates(obs, info, drone["pos"])
        obstacles = self._read_obstacles(obs, info, drone["pos"])

        return {
            "drone": drone,
            "gates": gates,
            "obstacles": obstacles,
            "sensor_range": self.sensor_range,
            "safety_limits": self.safety_limits,
        }

    def _read_drone_state(self, obs):
        pos = self._get_value(obs, ["pos", "position", "drone_pos"])
        vel = self._get_value(obs, ["vel", "velocity", "drone_vel"])
        rpy = self._get_value(obs, ["rpy", "attitude", "euler"])
        ang_vel = self._get_value(obs, ["ang_vel", "omega", "angular_velocity"])

        if pos is None:
            raise KeyError("Drone position not found in obs.")

        pos = np.array(pos, dtype=float)
        vel = np.zeros(3) if vel is None else np.array(vel, dtype=float)
        rpy = np.zeros(3) if rpy is None else np.array(rpy, dtype=float)
        ang_vel = np.zeros(3) if ang_vel is None else np.array(ang_vel, dtype=float)

        return {
            "pos": pos,
            "vel": vel,
            "rpy": rpy,
            "yaw": float(rpy[2]),
            "ang_vel": ang_vel,
        }

    def _read_gates(self, obs, info, drone_pos):
        gates_pos = self._get_value(obs, ["gates_pos", "gate_positions"])
        gates_quat = self._get_value(obs, ["gates_quat", "gate_quat"])

        if gates_pos is None and info is not None:
            gates_pos = self._get_value(info, ["gates_pos", "gate_positions"])
            gates_quat = self._get_value(info, ["gates_quat", "gate_quat"])

        if gates_pos is not None:
            return self._read_gates_from_pos_quat(gates_pos, gates_quat, drone_pos)

        gates_raw = self._get_value(obs, ["gates", "gate_poses"])

        if gates_raw is None and info is not None:
            gates_raw = self._get_value(info, ["gates", "gate_poses"])

        if gates_raw is None:
            gates_raw = self.nominal_gates

        return self._read_gates_from_generic(gates_raw, drone_pos)

    def _read_gates_from_pos_quat(self, gates_pos, gates_quat, drone_pos):
        gates = []
        gates_pos = np.asarray(gates_pos, dtype=float)

        if gates_quat is not None:
            gates_quat = np.asarray(gates_quat, dtype=float)

        for gate_id, pos in enumerate(gates_pos):
            if gates_quat is not None and gate_id < len(gates_quat):
                quat = gates_quat[gate_id]
                rot = R.from_quat(quat)

                rpy = rot.as_euler("xyz", degrees=False)
                yaw = float(rpy[2])

                normal = rot.apply([1.0, 0.0, 0.0])
                normal = normal / (np.linalg.norm(normal) + 1e-9)
            else:
                rpy, yaw, normal = self._nominal_gate_orientation(gate_id)

            distance = float(np.linalg.norm(pos - drone_pos))
            visible = distance <= self.sensor_range

            gates.append({
                "id": gate_id,
                "pos": pos.copy(),
                "rpy": rpy.copy(),
                "yaw": yaw,
                "normal": normal.copy(),
                "entry_dir": -normal.copy(),
                "exit_dir": normal.copy(),
                "distance": distance,
                "visible": visible,
                "source": "runtime",
            })

        return gates

    def _read_gates_from_generic(self, gates_raw, drone_pos):
        gates = []

        for gate_id, gate in enumerate(gates_raw):
            pos, rpy, yaw, normal = self._parse_gate(gate, gate_id)

            if pos is None:
                continue

            distance = float(np.linalg.norm(pos - drone_pos))
            visible = distance <= self.sensor_range

            gates.append({
                "id": gate_id,
                "pos": pos,
                "rpy": rpy,
                "yaw": yaw,
                "normal": normal,
                "entry_dir": -normal.copy(),
                "exit_dir": normal.copy(),
                "distance": distance,
                "visible": visible,
                "source": "runtime",
            })

        return gates

    def _parse_gate(self, gate, gate_id):
        pos = None
        rpy = None
        yaw = None
        quat = None

        if isinstance(gate, dict):
            pos = gate.get("pos", gate.get("position", None))
            rpy = gate.get("rpy", gate.get("attitude", None))
            yaw = gate.get("yaw", None)
            quat = gate.get("quat", gate.get("quaternion", None))
        else:
            pos = gate

        if pos is None:
            return None, None, None, None

        pos = np.array(pos, dtype=float)

        if quat is not None:
            rot = R.from_quat(quat)
            rpy = rot.as_euler("xyz", degrees=False)
            yaw = float(rpy[2])
            normal = rot.apply([1.0, 0.0, 0.0])
            normal = normal / (np.linalg.norm(normal) + 1e-9)
            return pos, rpy, yaw, normal

        if rpy is None:
            rpy, yaw, normal = self._nominal_gate_orientation(gate_id)
            return pos, rpy, yaw, normal

        rpy = np.array(rpy, dtype=float)

        if yaw is None:
            yaw = float(rpy[2])
        else:
            yaw = float(yaw)

        normal = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-9)

        return pos, rpy, yaw, normal

    def _nominal_gate_orientation(self, gate_id):
        if gate_id < len(self.nominal_gates):
            rpy = self.nominal_gates[gate_id]["rpy"].copy()
        else:
            rpy = np.zeros(3)

        yaw = float(rpy[2])
        normal = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-9)

        return rpy, yaw, normal

    def _read_obstacles(self, obs, info, drone_pos):
        obstacles_raw = self._get_value(obs, ["obstacles", "obstacle_positions", "obstacles_pos"])

        if obstacles_raw is None and info is not None:
            obstacles_raw = self._get_value(info, ["obstacles", "obstacle_positions", "obstacles_pos"])

        if obstacles_raw is None:
            obstacles_raw = self.nominal_obstacles

        obstacles = []

        for obstacle_id, obstacle in enumerate(obstacles_raw):
            if isinstance(obstacle, dict):
                pos = obstacle.get("pos", obstacle.get("position", None))
            else:
                pos = obstacle

            if pos is None:
                continue

            pos = np.array(pos, dtype=float)
            distance = float(np.linalg.norm(pos - drone_pos))
            visible = distance <= self.sensor_range

            obstacles.append({
                "id": obstacle_id,
                "pos": pos,
                "distance": distance,
                "visible": visible,
                "source": "runtime",
            })

        return obstacles

    def _get_value(self, data, possible_keys):
        if data is None:
            return None

        if isinstance(data, dict):
            for key in possible_keys:
                if key in data:
                    return data[key]

            for parent_key in ["drone", "state", "agent", "observation"]:
                if parent_key in data and isinstance(data[parent_key], dict):
                    for key in possible_keys:
                        if key in data[parent_key]:
                            return data[parent_key][key]

        return None