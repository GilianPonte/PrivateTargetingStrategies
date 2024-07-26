import tensorflow_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
import keras_tuner

def cnn(X, Y, T, scaling = True, batch_size = 100, epochs = 100, max_epochs = 10, folds = 5, directory = "tuner", seed = None):
  """
    Causal Neural Network (CNN) algorithm for estimating average treatment effects.

    Args:
        X (numpy.ndarray): Features matrix.
        Y (numpy.ndarray): Outcome vector.
        T (numpy.ndarray): Treatment vector.
        scaling (bool, optional): Whether to scale the features matrix. Default is True.
        batch_size (int, optional): Batch size for training. Default is 100.
        epochs (int, optional): Number of epochs for training. Default is 100.
        max_epochs (int, optional): Maximum number of epochs for hyperparameter optimization. Default is 10.
        folds (int, optional): Number of folds for cross-validation. Default is 5.
        directory (str, optional): Directory for saving hyperparameter optimization results. Default is "tuner".
        seed (int, optional): Seed for random number generation. Default is None.

    Returns:
        tuple: Tuple containing average treatment effect, CATE estimates, and trained tau_hat model.
    """

  import random
  import pandas as pd
  import tensorflow as tf
  from tensorflow import keras
  from keras import layers
  from sklearn.model_selection import KFold
  from sklearn.linear_model import LogisticRegression
  from keras.layers import Activation, LeakyReLU
  from keras import backend as K
  from keras.utils import get_custom_objects
  from sklearn.preprocessing import MinMaxScaler
  import numpy as np

  # set seeds
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  tf.keras.utils.set_random_seed(seed)

  # callback settings for early stopping and saving
  callback = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience = 5, mode = "min") # early stopping
  callback1 = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience = 5, mode = "min") # early stopping
  # define ate loss is equal to mean squared error between pseudo outcome and prediction of net.
  def ATE(y_true, y_pred):
    return tf.reduce_mean(y_pred, axis=-1)  # Note the `axis=-1`

  average_treatment_effect = []  # storage of ate estimates
  CATE_estimates = []

  ## scale the data for well-behaved gradients
  if scaling == True:
    scaler0 = MinMaxScaler(feature_range = (-1, 1))
    scaler0 = scaler0.fit(X)
    X = scaler0.transform(X)

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
    
  # not shuffling data here.
  X = np.array(pd.DataFrame(X))
  Y = np.array(pd.DataFrame(Y))
  T = np.array(pd.DataFrame(T))

  # save models
  checkpoint_filepath_mx = f"{directory}/m_x.hdf5"
  checkpoint_filepath_taux = f"{directory}/tau_x.hdf5"
  mx_callbacks = [callback, tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_mx, save_weights_only=False, monitor='val_loss', mode='min', save_freq="epoch", save_best_only=True),]
  tau_hat_callbacks = [callback, tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_taux, save_weights_only=False, monitor='val_loss', mode='min', save_freq="epoch", save_best_only=True),]

  y_tilde_hat = [] # collect all the \tilde{Y}
  T_tilde_hat = [] # collect all the \tilde{T}
  m_x_hat = [] # collect all m_x_hat for print
  e_x_hat = [] # collect all e_x_hat for print
    
  print("hyperparameter optimization for yhat")
  tuner = keras_tuner.Hyperband(
    hypermodel=build_model,
    objective="val_loss",
    max_epochs= max_epochs,
    overwrite=True,
    directory=directory,
    project_name="yhat",
    seed = seed,)
  tuner.search(X, Y, epochs = epochs, validation_split=0.25, verbose = 0, callbacks = [callback])

  # Get the optimal hyperparameters
  best_hps=tuner.get_best_hyperparameters()[0]
  print("the optimal architecture is: " + str(best_hps.values))
  
  cv = KFold(n_splits=folds, shuffle = False) # K-fold validation shuffle is off to prevent additional noise?

  for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
    # set random seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    
    #print("training model for m(x)")
    model_m_x = tuner.hypermodel.build(best_hps)
    model_m_x.fit(
      X[train_idx],
      Y[train_idx],
      epochs = epochs,
      batch_size = batch_size,
      validation_data = (X[test_idx], Y[test_idx]),
      callbacks= mx_callbacks, # prevent overfitting with early stopping.
      verbose = 0)
    model_m_x = tuner.hypermodel.build(best_hps)
    model_m_x.build(input_shape = (None,X.shape[1]))
    model_m_x.load_weights(checkpoint_filepath_mx)
    m_x = model_m_x.predict(x=X[test_idx], verbose = 0).reshape(len(Y[test_idx])) # obtain \hat{m}(x) from test set

    # obtain \tilde{Y} = Y_{i} - \hat{m}(x)
    truth = Y[test_idx].T.reshape(len(Y[test_idx]))
    y_tilde = truth - m_x
    y_tilde_hat = np.concatenate((y_tilde_hat,y_tilde)) # cbind in r
    m_x_hat = np.concatenate((m_x_hat,m_x)) # cbind in r

    ## fit \hat{e}(x)
    clf = LogisticRegression(verbose = 0).fit(X[train_idx], np.array(T[train_idx]).reshape(len(T[train_idx])))
    e_x = clf.predict_proba(X[test_idx]) # obtain \hat{e}(x)
    print(f"Fold {fold}: mean(m_x) = {np.round(np.mean(m_x), 2):.2f}, sd(m_x) = {np.round(np.std(m_x), 3):.3f} and mean(e_x) = {np.round(np.mean(e_x[:, 1]), 2):.2f}, sd(e_x) = {np.round(np.std(e_x[:, 1]), 3):.3f}")

    # obtain \tilde{T} = T_{i} - \hat{e}(x)
    #print("obtaining T_tilde")
    truth = T[test_idx].T.reshape(len(T[test_idx]))
    T_tilde = truth - e_x[:,1]
    T_tilde_hat = np.concatenate((T_tilde_hat,T_tilde))
    e_x_hat = np.concatenate((e_x_hat,e_x[:,1]))

  ## pseudo_outcome and weights
  pseudo_outcome = (y_tilde_hat/T_tilde_hat) # pseudo_outcome = \tilde{Y} / \tilde{T}
  w_weigths = np.square(T_tilde_hat) # \tilde{T}**2

    
  print("hyperparameter optimization for tau hat")
  tuner1 = keras_tuner.Hyperband(
    hypermodel=build_model,
    objective="val_loss",
    max_epochs=max_epochs,
    overwrite=True,
    directory=directory,
    project_name="tau_hat",
    seed = seed,)
  tuner1.search(X, pseudo_outcome, epochs=epochs, validation_split=0.25, verbose = 0, callbacks = [callback1])
  best_hps_tau =tuner1.get_best_hyperparameters()[0]
  print("the optimal architecture is: " + str(best_hps_tau.values))

  cv = KFold(n_splits=folds, shuffle = False)
  print("training for tau hat")
  for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
    # set random seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
      
    tau_hat = tuner1.hypermodel.build(best_hps_tau)
    history_tau = tau_hat.fit(
      X[train_idx],
      pseudo_outcome[train_idx],
      sample_weight= w_weigths[train_idx],
      epochs = epochs,
      batch_size = batch_size,
      callbacks = tau_hat_callbacks,
      validation_data = (X[test_idx], pseudo_outcome[test_idx]),
      verbose = 0)
    tau_hat = tuner1.hypermodel.build(best_hps_tau)
    tau_hat.build(input_shape = (None,X.shape[1]))
    tau_hat.load_weights(checkpoint_filepath_taux)
    CATE = tau_hat.predict(x=X[test_idx], verbose = 0).reshape(len(X[test_idx]))
    print(f"Fold {fold}: mean(tau_hat) = " + str(np.round(np.mean(CATE),2)) + ", sd(m_x) = " + str(np.round(np.std(CATE),3)))

    CATE_estimates = np.concatenate((CATE_estimates,CATE)) # store CATE's
  average_treatment_effect = np.append(average_treatment_effect, np.mean(CATE_estimates))
  print("ATE = " + str(np.round(np.mean(average_treatment_effect), 4)) + ", std(ATE) = " + str(np.round(np.std(average_treatment_effect), 3)))
  return average_treatment_effect, CATE_estimates, tau_hat

import tensorflow_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
import keras_tuner

def pcnn(X, Y, T, scaling=True, batch_size=100, epochs=100, max_epochs=1, directory="tuner", fixed_model = False, noise_multiplier=1, seed = 1):
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
    if (len(X)/2) % batch_size != 0:
        divisors = [i for i in range(1, int(math.sqrt((len(X)/2))) + 1) if (len(X)/2) % i == 0]
        divisors += [(len(X)/2) // i for i in divisors if (len(X)/2) // i != i]
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
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode="min")  # early stopping just like in rboost

    # define ate loss is equal to mean squared error between pseudo outcome and prediction of net.
    def ATE(y_true, y_pred):
        return tf.reduce_mean(y_pred, axis=-1)  # Note the `axis=-1`

    def generate_fixed_architecture(X):
      model = keras.Sequential()
      model.add(keras.Input(shape=(X.shape[1],)))
    
      # Define the architecture with 4 layers
      num_layers = 4
      units = 64
    
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

    cv = KFold(n_splits=2, shuffle=False)  # K-fold validation shuffle is off to prevent additional noise?

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
      tau_hat.compile(optimizer=tensorflow_privacy.DPKerasAdamOptimizer(l2_norm_clip=4, noise_multiplier=noise_multiplier, num_microbatches=batch_size, learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError(reduction=tf.losses.Reduction.NONE), metrics=[ATE]) # the microbatches are equal to the batch size. No microbatching applied.
      history_tau = tau_hat.fit(
        X[train_idx],
        pseudo_outcome[train_idx],
        sample_weight=w_weights[train_idx],
        epochs=epochs,
        batch_size=batch_size,
        callbacks=tau_hat_callbacks,
        validation_data=(X[test_idx], pseudo_outcome[test_idx]),
        verbose=0)
      if fixed_model == False:
        tau_hat = tuner.hypermodel.build(best_hps)
        tau_hat.build(input_shape=(None, X.shape[1]))
      tau_hat.load_weights(checkpoint_filepath_taux)
      CATE = tau_hat.predict(x=X[test_idx], verbose=0).reshape(len(X[test_idx]))
      print(f"Fold {fold}: mean(tau_hat) = {np.round(np.mean(CATE), 2):.2f}, sd(tau_hat) = {np.round(np.std(CATE), 3):.3f}")

      CATE_estimates = np.concatenate((CATE_estimates, CATE))  # store CATE's
    average_treatment_effect = np.mean(CATE_estimates)
    X = X[np.argsort(idx)]
    Y = Y[np.argsort(idx)]
    T = T[np.argsort(idx)]
    CATE_estimates = CATE_estimates[np.argsort(idx)]
    print(f"ATE = {average_treatment_effect}")    
    return average_treatment_effect, CATE_estimates, tau_hat, n, epsilon, noise_multiplier, epsilon_conservative
