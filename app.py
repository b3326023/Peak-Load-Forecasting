import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, LSTM
from keras.optimizers import Adam
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

raw_data = pd.read_csv('data/BasicData.csv')
data = raw_data['peak load(MW)']

def normalize(x):
  return (x - result_mean) / result_std

def unnormalize(x):
  return x * result_std + result_mean

# Set how many days should be use to predict single day
sequence_length = 30
result = []
for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])

# convert to numpy array
result = np.array(result, dtype='float64')

# save mean and standard deviation for normalization
result_mean = result.mean()
result_std = np.std(result)

# normalize the data
norm_result = normalize(result)

# split training and testing data
train_x, test_x, train_y, test_y = train_test_split(norm_result[:,:-1], norm_result[:,-1], test_size=0.2, random_state=0)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))


# build LSTM model and train the model
IN = Input(shape=(train_x.shape[1],1))
lstm1 = LSTM(10, return_sequences=True)(IN)
lstm2 = LSTM(15, return_sequences=False)(lstm1)
OUT = Dense(1, activation='linear')(lstm2)
model = Model(inputs=IN, outputs=OUT)
model.compile(loss="mse", optimizer="adam")
model.fit(train_x, train_y, epochs=100)

# Testing
predict = model.predict(test_x)
predict = np.reshape(predict, (predict.size,))

unnorm_predict = unnormalize(predict)
unnorm_test_y = unnormalize(test_y)

RMS = sqrt(mean_squared_error(unnorm_test_y, unnorm_predict))
print(RMS)

def predict_n_day(days, previous_data):
    previous_data = normalize(previous_data)
    for i in range(0, days):
      p = model.predict(np.expand_dims(previous_data[i : i+sequence_length], axis=0))
      previous_data = np.concatenate([previous_data, p])
    return unnormalize(previous_data[-days:])

previous_data = data[-sequence_length+1:].values
previous_data = np.expand_dims(previous_data, axis=1)
predicted = predict_n_day(days=7, previous_data=previous_data)
predicted = predicted.astype(int)

# Creat submission.csv
date = [i for i in range(20190402, 20190409)]
peak_load = predicted.flatten()
df = pd.DataFrame({'date':date, 'peak_load(MW)':peak_load})
df.to_csv('submission.csv', index=False)