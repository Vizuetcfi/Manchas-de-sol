# Librerias

import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Analisis exploratorio de datos

## Data
data_train = pd.read_csv('db/daily_sunspots_time_series_1850_2023.csv')
data_train.head()

## data info
data_train.info()

## data train
data_train.isnull().sum()

# Procesamiento de datos

## drop date and indicator column
data_train = data_train.drop(columns=['date', 'indicator'])
data_train.head()

## normalization
scaler = StandardScaler()
scaler.fit(data_train)
scaled_data = scaler.transform(data_train)
scaled_data = pd.DataFrame(scaled_data, columns=data_train.columns)
scaled_data.head()

## Series
series = scaled_data['counts'].values
time_step = scaled_data['year'].values


series
time_step

## Graficas
# plot sunspot counts over years
plt.figure(figsize=(14,7))
plt.plot(time_step, series)
plt.xlabel('Year')
plt.ylabel('Sunspot Counts')
plt.title('Sunspot Counts Over Years',
          fontsize=14)



# create function for windowed dataset
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    return ds.batch(batch_size).prefetch(1)

# split train and validation data where validation set is 20% of the total dataset
from sklearn.model_selection import train_test_split

time_train, x_train, time_valid, x_valid = train_test_split(series, 
                                                            time_step, 
                                                            test_size=0.2, 
                                                            shuffle=False)



# Hyperparameter
window_size = 30
batch_size = 32
shuffle_buffer_size = 5000

train_set = windowed_dataset(x_train, 
                             window_size=window_size, 
                             batch_size=batch_size, 
                             shuffle_buffer=shuffle_buffer_size)


# threshold mae callback (stop when mae < 10% of data scale)
class ThresholdedMAECallback(tf.keras.callbacks.Callback):
    def __init__(self , threshold):
        super(ThresholdedMAECallback, self).__init__()
        self.threshold = threshold
    
    def on_epoch_end(self, epoch, logs=None):
        current_mae = logs.get('mae')
        if current_mae < self.threshold:
            print(f"\nMAE has reached the threshold of ({self.threshold}), training is stopped.")
            self.model.stop_training = True

threshold_mae = (scaled_data['counts'].max() - scaled_data['counts'].min()) * 10/100
mae_callback = ThresholdedMAECallback(threshold_mae)

# Creacion de Modelos
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                        strides=1, padding='causal',
                        activation='relu', input_shape=[None, 1]),
  tf.keras.layers.LSTM(60, return_sequences=True),
  tf.keras.layers.Dense(30, activation='relu'),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1)
])


model.summary()

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)



history = model.fit(train_set, epochs=100, callbacks=[mae_callback])

# Plot training and validation loss per epoch
loss=history.history['loss']
epochs=range(len(loss))

plt.plot(epochs, loss, 'r')
plt.title('Training loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss"])

plt.figure()
