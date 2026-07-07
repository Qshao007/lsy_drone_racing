"""Small TOML reader for planner configuration values.

The planner modules use this wrapper instead of reading the TOML structure
directly. It keeps nominal gates, obstacles, sensor range, and safety limits in
one stable format.
"""

import tomllib

import numpy as np


class ConfigManager:
    """Expose the parts of an environment config needed by the planner."""

    def __init__(self, config_path):
        """Load a TOML config file once and keep the parsed dictionary."""
        with open(config_path, "rb") as f:
            self.cfg = tomllib.load(f)

    def get_sensor_range(self):
        """Return the gate/obstacle visibility radius."""
        return self.cfg["env"]["sensor_range"]

    def get_env_freq(self):
        """Return the high-level environment frequency."""
        return self.cfg["env"]["freq"]

    def get_control_mode(self):
        """Return whether the environment expects state or attitude commands."""
        return self.cfg["env"]["control_mode"]

    def get_nominal_gates(self):
        """Return nominal gate poses as numpy arrays."""
        gates = []
        for gate in self.cfg["env"]["track"]["gates"]:
            gates.append({
                "pos": np.array(gate["pos"], dtype=float),
                "rpy": np.array(gate["rpy"], dtype=float),
            })
        return gates

    def get_nominal_obstacles(self):
        """Return nominal obstacle positions as numpy arrays."""
        obstacles = []
        for obstacle in self.cfg["env"]["track"]["obstacles"]:
            obstacles.append({
                "pos": np.array(obstacle["pos"], dtype=float),
            })
        return obstacles

    def get_safety_limits(self):
        """Return configured world-position bounds."""
        safety = self.cfg["env"]["track"]["safety_limits"]
        return {
            "low": np.array(safety["pos_limit_low"], dtype=float),
            "high": np.array(safety["pos_limit_high"], dtype=float),
        }

    def is_track_randomized(self):
        """Return True when the track is randomized by the simulator."""
        return bool(self.cfg["env"]["track"]["randomize"])


if __name__ == "__main__":
    config_path = "config/level0.toml"

    cfg = ConfigManager(config_path)

    print("\n========== CONFIG DEBUG ==========")
    print("Config path:", config_path)

    print("\nSensor range:")
    print(cfg.get_sensor_range())

    print("\nEnv frequency:")
    print(cfg.get_env_freq())

    print("\nControl mode:")
    print(cfg.get_control_mode())

    print("\nTrack randomized:")
    print(cfg.is_track_randomized())

    print("\nNominal gates:")
    for i, gate in enumerate(cfg.get_nominal_gates()):
        print(f"Gate {i}: pos={gate['pos']}, rpy={gate['rpy']}")

    print("\nNominal obstacles:")
    for i, obstacle in enumerate(cfg.get_nominal_obstacles()):
        print(f"Obstacle {i}: pos={obstacle['pos']}")

    print("\nSafety limits:")
    safety = cfg.get_safety_limits()
    print("low :", safety["low"])
    print("high:", safety["high"])

    print("==================================\n")
