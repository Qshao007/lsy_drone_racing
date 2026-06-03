import numpy as np


class SelfLocation:
    def __init__(self):
        self.initialized = False
        self.origin_pos = np.zeros(3)
        self.origin_yaw = 0.0

        self.pos_world = np.zeros(3)
        self.vel_world = np.zeros(3)
        self.rpy_world = np.zeros(3)
        self.yaw_world = 0.0

    def update(self, parsed_obs):
        drone = parsed_obs["drone"]

        self.pos_world = np.array(drone["pos"], dtype=float)
        self.vel_world = np.array(drone["vel"], dtype=float)
        self.rpy_world = np.array(drone["rpy"], dtype=float)
        self.yaw_world = float(drone["yaw"])

        if not self.initialized:
            self.origin_pos = self.pos_world.copy()
            self.origin_yaw = self.yaw_world
            self.initialized = True

        return self.get_state()

    def get_state(self):
        return {
            "pos_world": self.pos_world.copy(),
            "vel_world": self.vel_world.copy(),
            "rpy_world": self.rpy_world.copy(),
            "yaw_world": self.yaw_world,
            "world_to_local": self.world_to_local,
            "local_to_world": self.local_to_world,
            "world_to_body": self.world_to_body,
            "body_to_world": self.body_to_world,
        }

    def world_to_local(self, pos_world):
        delta = np.array(pos_world, dtype=float) - self.origin_pos
        return self.rotation_matrix_z(self.origin_yaw).T @ delta

    def local_to_world(self, pos_local):
        return self.origin_pos + self.rotation_matrix_z(self.origin_yaw) @ np.array(pos_local, dtype=float)

    def world_to_body(self, pos_world):
        delta = np.array(pos_world, dtype=float) - self.pos_world
        return self.rotation_matrix_z(self.yaw_world).T @ delta

    def body_to_world(self, pos_body):
        return self.pos_world + self.rotation_matrix_z(self.yaw_world) @ np.array(pos_body, dtype=float)

    @staticmethod
    def rotation_matrix_z(yaw):
        c = np.cos(yaw)
        s = np.sin(yaw)

        return np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])