# from raw_observation import RawObservation


# def main():
#     # 1. Create fake observation
#     fake_obs = {
#         "pos": [-1.5, 0.75, 0.01],
#         "vel": [0.0, 0.0, 0.0],
#         "rpy": [0.0, 0.0, 0.0],
#         "ang_vel": [0.0, 0.0, 0.0],
#     }

#     # 2. Load raw observation parser
#     raw_obs = RawObservation(
#         config_path="config/level0.toml"
#     )

#     # 3. Parse fake observation
#     parsed = raw_obs.update(fake_obs)

#     # 4. Print result
#     raw_obs.print_summary(parsed)


# if __name__ == "__main__":
#     main()




# test_self_location.py

# from raw_observation import RawObservation
# from self_location import SelfLocation


# def main():
#     raw_obs = RawObservation(
#         config_path="config/level0.toml"
#     )

#     self_loc = SelfLocation()

#     fake_obs_1 = {
#         "pos": [-1.5, 0.75, 0.01],
#         "vel": [0.0, 0.0, 0.0],
#         "rpy": [0.0, 0.0, 0.0],
#     }

#     parsed_1 = raw_obs.update(fake_obs_1)
#     self_loc.update(parsed_1)
#     self_loc.print_summary()

#     fake_obs_2 = {
#         "pos": [-1.0, 0.75, 0.70],
#         "vel": [0.5, 0.0, 0.0],
#         "rpy": [0.0, 0.0, 0.0],
#     }

#     parsed_2 = raw_obs.update(fake_obs_2)
#     state = self_loc.update(parsed_2)
#     self_loc.print_summary()

#     print("Returned state:")
#     print(state)

#     gate_world = [0.5, 0.25, 0.7]
#     gate_local = self_loc.world_to_local(gate_world)

#     print("\nGate world:", gate_world)
#     print("Gate local:", gate_local)


# if __name__ == "__main__":
#     main()

# test_local_mapping.py

# from config_manager import ConfigManager
# from raw_observation import RawObservation
# from self_location import SelfLocation
# from local_mapping import LocalMapping


# def main():
#     config_path = "config/level0.toml"

#     config = ConfigManager(config_path)
#     raw_obs = RawObservation(config_path)
#     self_loc = SelfLocation()
#     local_map = LocalMapping(config)

#     fake_obs = {
#         "pos": [-1.5, 0.75, 0.01],
#         "vel": [0.0, 0.0, 0.0],
#         "rpy": [0.0, 0.0, 0.0],
#     }

#     parsed = raw_obs.update(fake_obs)
#     self_loc.update(parsed)

#     estimated_map = local_map.update(parsed, self_loc)
#     local_map.print_summary(estimated_map)


# if __name__ == "__main__":
#     main()


# test_planner.py

# from config_manager import ConfigManager
# from raw_observation import RawObservation
# from self_location import SelfLocation
# from local_mapping import LocalMapping
# from planner import SimpleGatePlanner


# def main():
#     config_path = "config/level0.toml"

#     config = ConfigManager(config_path)
#     raw_obs = RawObservation(config_path)
#     self_loc = SelfLocation()
#     local_map = LocalMapping(config)
#     planner = SimpleGatePlanner(pass_threshold=0.35)

#     fake_obs = {
#         "pos": [-1.5, 0.75, 0.01],
#         "vel": [0.0, 0.0, 0.0],
#         "rpy": [0.0, 0.0, 0.0],
#     }

#     parsed = raw_obs.update(fake_obs)
#     self_state = self_loc.update(parsed)
#     estimated_map = local_map.update(parsed, self_loc)

#     target = planner.update(estimated_map, self_state)
#     planner.print_summary(target)


# if __name__ == "__main__":
#     main()


# test_trajectory_manager.py

from config_manager import ConfigManager
from raw_observation import RawObservation
from self_location import SelfLocation
from local_mapping import LocalMapping
from trajectory_manager import TrajectoryManager


def main():
    config_path = "config/level0.toml"

    config = ConfigManager(config_path)
    raw_obs = RawObservation(config_path)
    self_loc = SelfLocation()
    local_map = LocalMapping(config)

    trajectory = TrajectoryManager(
        waypoint_threshold=0.25,
        gate_offset=0.35,
        max_speed=0.8,
        loop=True,
    )

    fake_obs = {
        "pos": [-1.5, 0.75, 0.01],
        "vel": [0.0, 0.0, 0.0],
        "rpy": [0.0, 0.0, 0.0],
    }

    parsed = raw_obs.update(fake_obs)
    self_state = self_loc.update(parsed)
    estimated_map = local_map.update(parsed, self_loc)

    ref = trajectory.update(self_state, estimated_map)

    trajectory.print_summary(ref)

    print("Number of waypoints:", len(trajectory.waypoints))

    for i, wp in enumerate(trajectory.waypoints):
        print(
            f"WP {i}: "
            f"type={wp['type']}, "
            f"gate={wp['gate_id']}, "
            f"pos={wp['pos']}"
        )


if __name__ == "__main__":
    main()