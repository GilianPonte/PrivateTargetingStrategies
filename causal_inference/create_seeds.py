import random
import numpy as np
def generate_and_write_seeds(file_path, num_seeds, seed=None):
    # If a seed is provided, use it for reproducibility
    if seed is not None:
        random.seed(seed)

    # Generate random seeds deterministically
    seeds = [random.getrandbits(32) for _ in range(num_seeds)]

    # Write seeds to a text file
    with open(file_path, "w") as file:
        for seed in seeds:
            file.write(str(seed) + "\n")

    print("Seeds have been written to", file_path)

def read_file(file_path):
    with open(file_path, "r") as file:
        return [int(seed.strip()) for seed in file.readlines()]

# Set the initial seed for reproducibility
initial_seed = 422312

# Generate and write seeds for seeds_data.txt
generate_and_write_seeds("seeds_data.txt", 100, seed=initial_seed)
generate_and_write_seeds("seeds_training.txt", 700, seed=initial_seed)
