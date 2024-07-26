
def pcnn(X, Y, T, scaling=True, batch_size=100, epochs=100, max_epochs=1, directory="tuner", C =4, fixed_model = False, noise_multiplier=1, seed = 1):
    """
    Private Causal Neural Network (PCNN) algorithm for estimating average treatment effects.

    Args:
    X (numpy.ndarray): Features matrix.
    Y (numpy.ndarray): Outcome vector.
    T (numpy.ndarray): Treatment vector.
    scaling (bool, optional): Whether to scale the features matrix. Default is True.
    simulations (int, optional): Number of simulations. Default is 1.
    batch_size (int, optional): Batch size for training. Default is 100.
    epochs (int, optional): Number of epochs for training. Default is 100.
    max_epochs (int, optional): Maximum number of epochs for hyperparameter optimization. Default is 10.
    directory (str, optional): Directory for saving hyperparameter optimization results. Default is "tuner".
    noise_multiplier (float, optional): Noise multiplier for differential privacy. Default is 1.

    Returns:
    tuple: Tuple containing average treatment effect, CATE estimates, trained tau_hat model, and privacy risk .
    """

    import random
    import re
    import numpy as np
    import tensorflow as tf
    import tensorflow_privacy
    from tensorflow import keras
    from keras.layers import Activation, LeakyReLU
    from keras import backend as K
    from keras import layers
    from keras.utils import get_custom_objects
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    import math
    import keras_tuner

    # set random seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    # Check if batch size divides the data evenly
    if len(X) % batch_size != 0:
        divisors = [i for i in range(1, int(math.sqrt(len(X))) + 1) if len(X) % i == 0]
        divisors += [len(X) // i for i in divisors if len(X) // i != i]
        divisors.sort()
        raise ValueError("The batch size you have specified does not divide the data into a whole number.\nPlease select one of the following possible batch sizes: {}".format(np.round(divisors)))

    # Calculate epsilon
    statement = tensorflow_privacy.compute_dp_sgd_privacy_statement(
        number_of_examples=len(X),
        batch_size=batch_size,
        num_epochs=epochs,
        noise_multiplier=noise_multiplier,
        delta=1/len(X),
        used_microbatching=False,
        max_examples_per_user=1
    )
    print(statement)

    # Extract epsilon and noise_multiplier from the statement
    numbers = [float(num) if '.' in num else int(num) for num in re.findall(r'\d+\.\d+|\d+', statement)]
    n, epsilon, noise_multiplier, epsilon_conservative = numbers[0], numbers[8], numbers[2], numbers[7]


    # callback settings for early stopping and saving
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode="min")  # early stopping just like in rboost

    # define ate loss is equal to mean squared error between pseudo outcome and prediction of net.
    def ATE(y_true, y_pred):
        return tf.reduce_mean(y_pred, axis=-1)  # Note the `axis=-1`

    def generate_fixed_architecture(X):
      model = keras.Sequential()
      model.add(keras.Input(shape=(X.shape[1],)))

      # Define the architecture with 4 layers
      num_layers = 4
      units = 512

      for _ in range(num_layers):
          model.add(layers.Dense(units, activation='tanh')) # https://arxiv.org/pdf/2007.14191.pdf
          units = max(units // 2, 1)  # Reduce the number of units by half for each subsequent layer

      # Add output layer
      model.add(layers.Dense(1, activation='linear'))

      model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["MSE"],
      )
      return model

    average_treatment_effect = []  # storage of ate estimates
    all_CATE_estimates = []  # Store CATE estimates for each simulation

    ## scale the data for well-behaved gradients
    if scaling == True:
        scaler0 = MinMaxScaler(feature_range=(-1, 1))
        scaler0 = scaler0.fit(X)
        X = scaler0.transform(X)
        X = pd.DataFrame(X)

    ## Add leaky-relu so we can use it as a string
    get_custom_objects().update({'leaky-relu': Activation(LeakyReLU(alpha=0.2))})

    def build_model(hp):
        model = keras.Sequential()
        model.add(keras.Input(shape=(X.shape[1],)))
        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 1, 4)):
            model.add(
                layers.Dense(
                    # Tune number of units separately.
                    units=hp.Choice(f"units_{i}", [8, 16, 32, 64, 256, 512]),
                    activation=hp.Choice("activation", ["leaky-relu", "relu"]),
                )
            )
        model.add(layers.Dense(1, activation="linear"))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mean_squared_error",
            metrics=["MSE"],
        )
        return model

    # for epsilon calculation. Shuffeling serves as a Poisson sampling substitute (Ponomareva et al. 2023)
    idx = np.random.permutation(pd.DataFrame(X).index)
    X = np.array(pd.DataFrame(X).reindex(idx))
    Y = np.array(pd.DataFrame(Y).reindex(idx))
    T = np.array(pd.DataFrame(T).reindex(idx))

    # save models
    checkpoint_filepath_mx = f"{directory}/_{epsilon}_m_x.hdf5"
    checkpoint_filepath_taux = f"{directory}/_{epsilon}_tau_x.hdf5"

    mx_callbacks = [callback,
      tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_mx, save_weights_only=False, monitor='val_loss', mode='min', save_freq="epoch", save_best_only=True,)]
    tau_hat_callbacks = [callback,
      tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_taux, save_weights_only=False, monitor='val_loss', mode='min', save_freq="epoch", save_best_only=True,)]
    y_tilde_hat = []  # collect all the \tilde{Y}
    T_tilde_hat = []  # collect all the \tilde{T}
    m_x_hat = []  # collect all m_x_hat for print
    e_x_hat = []  # collect all e_x_hat for print

    print("hyperparameter optimization for yhat")
    tuner = keras_tuner.Hyperband(hypermodel=build_model, objective="val_loss", max_epochs=max_epochs, overwrite=True, directory=directory, project_name="yhat",seed=seed,) # random search is at least as slow..

    tuner.search(X, Y, epochs=epochs, validation_split=0.25, verbose=0, callbacks=[mx_callbacks])
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters()[0]
    print("the optimal architecture is: " + str(best_hps.values))

    cv = KFold(n_splits=2, shuffle=False)  # K-fold validation

    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
      # set random seeds
      np.random.seed(seed)
      tf.random.set_seed(seed)
      random.seed(seed)
      tf.keras.utils.set_random_seed(seed)

      # training model for m(x)
      model_m_x = tuner.hypermodel.build(best_hps)
      model_m_x.fit(X[train_idx],
                    Y[train_idx],
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X[test_idx], Y[test_idx]),
                    callbacks=mx_callbacks,
                    verbose=0)
      model_m_x = tuner.hypermodel.build(best_hps)
      model_m_x.build(input_shape=(None, X.shape[1]))
      model_m_x.load_weights(checkpoint_filepath_mx)
      m_x = model_m_x.predict(x=X[test_idx], verbose=0).reshape(len(Y[test_idx]))  # obtain \hat{m}(x) from test set

      # obtain \tilde{Y} = Y_{i} - \hat{m}(x)
      truth = Y[test_idx].T.reshape(len(Y[test_idx]))
      y_tilde = truth - m_x
      y_tilde_hat = np.concatenate((y_tilde_hat, y_tilde))  # cbind in r
      m_x_hat = np.concatenate((m_x_hat, m_x))  # cbind in r

      # fit \hat{e}(x)
      clf = LogisticRegression(verbose=0).fit(X[train_idx], np.array(T[train_idx]).reshape(len(T[train_idx])))
      e_x = clf.predict_proba(X[test_idx])  # obtain \hat{e}(x)
      print(f"Fold {fold}: mean(m_x) = {np.round(np.mean(m_x), 2):.2f}, sd(m_x) = {np.round(np.std(m_x), 3):.3f} and mean(e_x) = {np.round(np.mean(e_x[:, 1]), 2):.2f}, sd(e_x) = {np.round(np.std(e_x[:, 1]), 3):.3f}")

      # obtain \tilde{T} = T_{i} - \hat{e}(x)
      truth = T[test_idx].T.reshape(len(T[test_idx]))
      T_tilde = truth - e_x[:, 1]
      T_tilde_hat = np.concatenate((T_tilde_hat, T_tilde))
      e_x_hat = np.concatenate((e_x_hat, e_x[:, 1]))

    # storage
    CATE_estimates = []

    # pseudo_outcome and weights
    pseudo_outcome = (y_tilde_hat / T_tilde_hat)  # pseudo_outcome = \tilde{Y} / \tilde{T}
    w_weights = np.square(T_tilde_hat)  # \tilde{T}**2

    cv = KFold(n_splits=2, shuffle=False)

    print("training for tau hat")
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
      # set random seeds
      np.random.seed(seed)
      tf.random.set_seed(seed)
      random.seed(seed)
      tf.keras.utils.set_random_seed(seed)

      tau_hat = generate_fixed_architecture(X) # an alternative is to fix the values of hyperparameters to some reasonable defaults and forgo hyperparameter tuning altogether (Ponomareva et al. 2023)
      if fixed_model == False:
        tau_hat = tuner.hypermodel.build(best_hps)
      tau_hat.compile(optimizer=tensorflow_privacy.DPKerasAdamOptimizer(l2_norm_clip=C, noise_multiplier=noise_multiplier, num_microbatches=batch_size, learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError(reduction=tf.losses.Reduction.NONE), metrics=[ATE]) # the microbatches are equal to the batch size. No microbatching applied.
      history_tau = tau_hat.fit(
        X[train_idx],
        pseudo_outcome[train_idx],
        sample_weight=w_weights[train_idx],
        epochs=epochs,
        batch_size=batch_size,
        callbacks=tau_hat_callbacks,
        validation_data=(X[test_idx], pseudo_outcome[test_idx]),
        verbose=1)
      CATE = tau_hat.predict(x=X[test_idx], verbose=0).reshape(len(X[test_idx]))
      print(f"Fold {fold}: mean(tau_hat) = {np.round(np.mean(CATE), 2):.2f}, sd(tau_hat) = {np.round(np.std(CATE), 3):.3f}")

      CATE_estimates = np.concatenate((CATE_estimates, CATE))  # store CATE's
    average_treatment_effect = np.mean(CATE_estimates)
    print(f"ATE = {average_treatment_effect}")
    return average_treatment_effect, CATE_estimates, tau_hat, n, epsilon, noise_multiplier, epsilon_conservative, pseudo_outcome

!pip install git+https://github.com/GilianPonte/PrivateTargeting.git -q

import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import causal_inference
from causal_inference import simulation_data
from causal_inference import strategy1
import time
from google.colab import drive
tf.config.experimental.enable_op_determinism()
#drive.mount('/content/drive')

start_time = time.time()

seed = 1
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)

dataframe = pd.read_csv('data.csv', sep = ',', header = 0)
X = dataframe.iloc[:,1:16] # features
T = dataframe.iloc[:,16]
Y = dataframe.iloc[:,17]

average_treatment_effect, CATE_estimates, tau_hat = strategy1.cnn(X = X,
                                                                  Y = Y,
                                                                  T = T,
                                                                  scaling = True,
                                                                  batch_size = 100,
                                                                  epochs = 100,
                                                                  max_epochs = 10,
                                                                  folds = 10,
                                                                  directory = "tuner",
                                                                  seed = seed)
np.savetxt("CATE_estimates.csv", CATE_estimates, delimiter=",")

X_out = pd.read_csv('/content/X_out.csv')
CATE_estimates_out = tau_hat.predict(X_out)

"""# causal forest"""

!pip install econml -q
from econml.dml import CausalForestDML
# Define the model
causal_forest = CausalForestDML(n_estimators=100,
                                model_y = "forest",
                                model_t = "forest",
                                discrete_treatment = True,
                                random_state=1)


# Fit the model
causal_forest.fit(Y, T, X=X)

# Estimate the Conditional Average Treatment Effects (CATE)
cate_estimates = causal_forest.effect(X)

# Show the estimated CATE
print(cate_estimates)

from econml.dml import DML
from sklearn.ensemble import GradientBoostingRegressor

# Initialize the model with Gradient Boosting for both outcome and treatment models
model = DML(model_y=GradientBoostingRegressor(),
            model_t=GradientBoostingRegressor(),
            model_final=GradientBoostingRegressor(),
            random_state = 1)  # k-folds cross-validation

# Fit the model
model.fit(Y, T, X)

# Predict the Conditional Average Treatment Effects (CATE)
cate_predictions_boost = model.effect(X)

# Calculate and print the mean of the CATE predictions
mean_cate = np.mean(cate_predictions_boost)
print("Mean CATE:", mean_cate)

from econml.dml import LinearDML
from sklearn.linear_model import LassoCV

# Initialize the model with Lasso for the outcome and treatment
model = LinearDML(
    model_y='linear',   # Using Lasso with cross-validation for the outcome model
    model_t='linear',   # Using Lasso with cross-validation for the treatment model
    random_state=1  # Ensuring reproducibility
)

# Fit the model
model.fit(Y, T, X)

# Predict the Conditional Average Treatment Effects (CATE)
cate_predictions_lasso = model.effect(X)
# Calculate and print the mean of the CATE predictions
mean_cate = np.mean(cate_predictions_lasso)
print("Mean CATE:", mean_cate)

"""# privacy

"""

!pip install git+https://github.com/GilianPonte/PrivateTargeting.git -q
import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import causal_inference
from causal_inference import strategy1
import time


start_time = time.time()

seed = 2132131
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)

dataframe = pd.read_csv('data.csv', sep = ',', header = 0)
X = dataframe.iloc[:,1:16] # features
T = dataframe.iloc[:,16]
Y = dataframe.iloc[:,17]

results_list = []

# noise multipliers
noise_multipliers = [0.14] #76.9,21,2.67,1.51,0.797,0.5184

for noise_multiplier in noise_multipliers:
  random.seed(seed)
  tf.random.set_seed(seed)
  np.random.seed(seed)
  tf.keras.utils.set_random_seed(seed)

  # Define the directory based on the noise multiplier
  directory = f"tuner_{noise_multiplier}"
  os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist

  # Call the function with the current noise_multiplier value
  average_treatment_effect, CATE_estimates, tau_hat, n, epsilon, noise_multiplier, epsilon_conservative,pseudo_outcome = pcnn(
        X=X,
        Y=Y,
        T=T,
        scaling=True,
        batch_size= 341,
        C = 4,
        epochs=100,
        max_epochs=10,
        fixed_model = False,
        directory=directory,  # Use the directory variable here
        noise_multiplier=noise_multiplier,
        seed = seed
        )

  # Append the results to the list
  results_list.append({
      'Noise Multiplier': noise_multiplier,
      'Average Treatment Effect': average_treatment_effect,
      'CATE Estimates': CATE_estimates,
      'Epsilon': epsilon
      })
  end_time = time.time()
  execution_time_one_sim = end_time - start_time
  print("Execution time one sim: {:.2f} seconds".format(execution_time_one_sim))

# Print or use the DataFrames as needed
print(results_list)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time one sim: {:.2f} seconds".format(execution_time))

def summary_statistics(data):
    import numpy as np
    import matplotlib.pyplot as plt
    stats = {
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Standard Deviation': np.std(data),
        'Minimum': np.min(data),
        'Maximum': np.max(data),
        'Sum': np.sum(data),
        'Count': len(data),
        'Variance': np.var(data)
    }
    # Plotting the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins='auto', color='blue', alpha=0.7)
    plt.title('Histogram of Data')
    plt.xlabel('Data Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    return stats

summary_statistics(pseudo_outcome)

import tensorflow_privacy
import math
import numpy as np
noise_multipliers = [76.9,21,2.67,1.51,0.797, 0.5184]

for noise_multiplier in noise_multipliers:
  epsilon = np.round(tensorflow_privacy.compute_dp_sgd_privacy(n=(178343*2), batch_size=341, noise_multiplier=noise_multiplier, epochs=100, delta=1/(178343*2))[0], 2)
  print("epsilon  = " + str(epsilon) + ", the privacy risk increases with " + str(np.round((math.exp(epsilon)-1)*100, 2)) + " percent")
