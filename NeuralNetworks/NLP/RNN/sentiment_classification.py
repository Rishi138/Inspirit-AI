import pandas as pd  # Loading Data
import keras  # Build Model and Tokenization
from sklearn.model_selection import train_test_split  # Split our data into training and testing data
from sklearn.metrics import accuracy_score  # See the accuracy of our model

# Loading Data
dataframe = pd.read_csv("yelp_final.csv")

# Creating our input which will be the review
X_data = dataframe["text"].values

# If a review is 4 or 5 stars classify it as positive with a 1
data = []
for x in dataframe["stars"]:
    if x > 3:
        data.append(1)
    else:
        data.append(0)
# Creating our labels which is positive or negative
dataframe["sentiment"] = data
y_data = dataframe["sentiment"].values
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
# Building Model
model = keras.Sequential()
# Input layer
model.add(keras.Input(shape=(1,), dtype="string"))

# Making a layer to vectorize our string
vectorize_layer = keras.layers.TextVectorization(
    max_tokens=5000,
    output_mode='int',
    output_sequence_length=100
)

# Call adapt(), which fits the TextVectorization layer to our text dataset.
# This is when the max_tokens most common words (i.e. the vocabulary) are selected.
# It goes through our data to create a vocabulary and tokenizes our text
# It converts our text into numerical representations
vectorize_layer.adapt(X_train)

# Adding layer to our model
model.add(vectorize_layer)

# Our next layer will be an Embedding layer, which will turn the integers produced by the previous layer into
# fixed-length vectors. Note that we're using max_tokens + 1 here, since there's an out-of-vocabulary (OOV) token
# that gets added to the vocab.
# 128 different attributes
model.add(keras.layers.Embedding(5001, 128))

# Adding our LSTM (Long short-term Memory) layer is our recurrent layer which is what makes this an RNN (Recurrent
# Neural Network) 50 is the dimensionality of the output space Simply, put it is the amount of features The LSTM is a
# form of feature extraction
# uses 128 attributes to find 50 patterns
# then dense layers use the 50 patterns to predict
model.add(keras.layers.LSTM(50, return_sequences=True))
# return sequences make sure we return a 3d array since they expect the input data in a 3d array format
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.LSTM(50))
model.add(keras.layers.Dropout(0.5))

# Adding two fully connected dense layers
# These dense layers now use the patterns found from the LSTM to make predictions
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

# We now need to compile and create the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training our model
model.fit(X_train, y_train, epochs=10)

y_pred = model.predict(X_test)
y_pred = (y_pred >= 0.5).astype(int)
print(accuracy_score(y_pred, y_test))
