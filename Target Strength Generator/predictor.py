from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers



def predictor(norm):
  model = keras.Sequential([
      norm,
      (layers.Dense(64, activation='relu')),
      (layers.Dense(32, activation='relu')),
      (layers.Dense(16, activation='relu')),
      (layers.Dense(1))
  ])

  model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model



