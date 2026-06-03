import tomllib
import numpy as np


class ConfigManager:
    def __init__(self, config_path):
        with open(config_path, "rb") as f:
            self.cfg = tomllib.load(f)

    def get_sensor_range(self):
        return self.cfg["env"]["sensor_range"]

    def get_env_freq(self):
        return self.cfg["env"]["freq"]

    def get_control_mode(self):
        return self.cfg["env"]["control_mode"]

    def get_nominal_gates(self):
        gates = []
        for gate in self.cfg["env"]["track"]["gates"]:
            gates.append({
                "pos": np.array(gate["pos"], dtype=float),
                "rpy": np.array(gate["rpy"], dtype=float),
            })
        return gates

    def get_nominal_obstacles(self):
        obstacles = []
        for obstacle in self.cfg["env"]["track"]["obstacles"]:
            obstacles.append({
                "pos": np.array(obstacle["pos"], dtype=float),
            })
        return obstacles

    def get_safety_limits(self):
        safety = self.cfg["env"]["track"]["safety_limits"]
        return {
            "low": np.array(safety["pos_limit_low"], dtype=float),
            "high": np.array(safety["pos_limit_high"], dtype=float),
        }

    def is_track_randomized(self):
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