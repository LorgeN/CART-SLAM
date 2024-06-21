import json
from copy import deepcopy

base_config = [
    {"type": "disparity", "smoothing_radius": 3, "smoothing_iterations": 1},
    {"type": "disparity_derivative"},
    {"type": "superpixels", "iterations": 8, "block_size": 10},
    {"type": "superpixels_visualization"},
]


def generate_configs(var_to_change: str, values: list):
    for value in values:
        config = deepcopy(base_config)

        for module in config:
            if var_to_change in module:
                module[var_to_change] = value

        with open(f"kitti-{var_to_change}-{value}.json", "w") as f:
            json.dump(config, f, indent=4)


values_to_change = [
    ("iterations", [2, 4, 6, 8, 16, 32]),
    ("block_size", [8, 10, 12, 16, 18, 20, 40]),
]

if __name__ == "__main__":
    for var, values in values_to_change:
        generate_configs(var, values)
