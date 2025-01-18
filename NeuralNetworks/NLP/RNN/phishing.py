import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataframe = pd.read_csv("phishing.csv")
dataframe.dropna(inplace=True)

y_data = dataframe["label"]
X_data = dataframe["body"]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

model = keras.Sequential()
model.add(keras.layers.InputLayer(shape=(1,), dtype="string"))
vectorize_layer = keras.layers.TextVectorization(
    max_tokens=5000,
    output_mode="int",
    output_sequence_length=100
)
vectorize_layer.adapt(X_train)
model.add(vectorize_layer)
model.add(keras.layers.Embedding(5001, 128))
model.add(keras.layers.LSTM(50, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.LSTM(50))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=1)
json_string = model.to_json()
with open('rnn_phishing_model.json', 'w') as f:
    f.write(json_string)
y_pred = model.predict(X_test)
y_pred = (y_pred >= 0.5).astype(int)
print(accuracy_score(y_pred, y_test))

# 2kFTsiFtOAR63ZvYivgueOGQbXK_5xGAp1JzGCBKi8MBRAbEH
