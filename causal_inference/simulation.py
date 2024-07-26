# install PCNNs and simulation setup
!pip install git+https://github.com/GilianPonte/PrivateTargeting.git -q

import os
import pandas as pd
import numpy as np
import random
import tensorflow
import causal_inference
from causal_inference import simulation_data
from causal_inference import strategy1
import time

try:
    seeds_data = np.genfromtxt('/content/seeds_data.txt', delimiter=',', dtype = np.int64)
except IOError:
    print("Error: File not found or could not be read.")
try:
    seeds_training = np.genfromtxt('/content/seeds_training.txt', delimiter=',', dtype = np.int64)
except IOError:
    print("Error: File not found or could not be read.")

# set time
start_time = time.time()
tensorflow.config.experimental.enable_op_determinism()
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Read seeds_data and seeds_training from file
seeds_data = read_file("seeds_data.txt")
seeds_training = read_file("seeds_training.txt")

# simulation parameters
iterations = 1
results_list = []
noise_multipliers = [0] # Initialize lists to store results for each noise multiplier ,8.7,3.4,1.12,0.845,0.567,0.3543

for i in range(iterations):
  print("Iteration: {}".format(i, i))
  random.seed(seeds_data[i])
  tensorflow.random.set_seed(seeds_data[i])
  np.random.seed(seeds_data[i])
  tensorflow.keras.utils.set_random_seed(seeds_data[i])

  # read data
  data = simulation_data.data_simulation(100000)

  # Separate the columns into different DataFrames
  x = data[['covariate_1', 'covariate_2', 'covariate_3', 'covariate_4', 'covariate_5', 'covariate_6']]
  w, m, tau, mu1, mu0, y = data[['w']], data[['m']], data[['tau']], data[['mu1']], data[['mu0']], data[['y']]

  # Loop through each noise multiplier value
  for noise_index, noise_multiplier in enumerate(noise_multipliers):
    print(noise_multiplier)

    for a in range(1):
      combined_number = (noise_index * 100) + a
      print("Combined number: {}".format(combined_number, combined_number))
      random.seed(seeds_training[combined_number])
      tensorflow.random.set_seed(seeds_training[combined_number])
      np.random.seed(seeds_training[combined_number])
      tensorflow.keras.utils.set_random_seed(seeds_training[combined_number])

      # Define the directory based on the noise multiplier
      directory = f"tuner_{noise_multiplier}_iteration_{i}_algo_run_{a}"
      os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist

      # Call the function with the current noise_multiplier value
      if noise_multiplier == 0:
        print("no privacy")
        average_treatment_effect, CATE_estimates, tau_hat = strategy1.cnn(X = x,
                                                                  Y = y,
                                                                  T = w,
                                                                  scaling = True,
                                                                  batch_size = 100,
                                                                  epochs = 100,
                                                                  max_epochs = 10,
                                                                  folds = 10,
                                                                  directory = directory,
                                                                  seed = seeds_training[combined_number])
        epsilon = 0
        n = len(x)
        epsilon_conservative = 0
      if noise_multiplier != 0:
        average_treatment_effect, CATE_estimates, tau_hat, n, epsilon, noise_multiplier, epsilon_conservative = strategy1.pcnn(
            X=x,
            Y=y,
            T=w,
            scaling=True,
            batch_size=100,
            epochs=100,
            max_epochs=10,
            fixed_model = True,
            directory=directory,  # Use the directory variable here
            noise_multiplier=noise_multiplier,
            seed = seeds_training[combined_number]
            )
      # Append the results to the list
      results_list.append({
          'Noise Multiplier': noise_multiplier,
          'Average Treatment Effect': average_treatment_effect,
          'CATE Estimates': CATE_estimates,
          'Epsilon': epsilon,
          'epsilon conservative': epsilon_conservative,
          'true ate': data[['tau']].mean(),
          'true CATE': data[['tau']],
          'sample size': n,
          'data set' : i,
          'iteration' : a,
          'covariates' : x,
          'y' : y,
          'w' : w,
          })
      
      np.save(f"results_list_{noise_multiplier}", results_list)

      # Print or use the DataFrames as needed
      print(results_list)
      end_time = time.time()
      execution_time = end_time - start_time
      print("Execution time one sim: {:.2f} seconds".format(execution_time))
