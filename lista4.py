import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

input_data = pd.read_csv('radio_train/input_data.csv', header=None)
target_data = pd.read_csv('radio_train/target_data.csv', header=None).iloc[:, 0:2]
train_data, test_data, train_target, test_target = train_test_split(input_data, target_data, test_size=0.2, random_state=42)

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.fit_transform(test_data)
train_target = scaler.fit_transform(train_target)
test_target = scaler.fit_transform(test_target)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, input_shape=(16,), activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')
])

model.compile(loss=tf.keras.losses.MeanSquaredError(), metrics='accuracy', optimizer=tf.optimizers.Adam(learning_rate=0.01, use_ema=True, ema_momentum=0.5))
model.fit(train_data, train_target, epochs=50, batch_size=32)
prediction_score = model.evaluate(test_data, test_target)
print(prediction_score)