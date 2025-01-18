import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Loading data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Plot first image
plt.imshow(X_train[0])
plt.show()

# Checking shape
print(X_train.shape)

# Reshape data
# 60000 images with each image having dimensions 28 x 28
# The 1 makes it grayscale
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# One-hot encoding of labels
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
print(y_train[0])

# Creating model
model = keras.models.Sequential()
# Input model
model.add(keras.layers.Input((28, 28, 1)))
# Convolutional layers for feature extraction
# Use kernels or filters of size 3x3 to extract features
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
# Find 32 new features
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
# Converts 2d array in 1d vector
model.add(keras.layers.Flatten())
# Fully connected layer that returns probabilities for 10 different possibilities
model.add(keras.layers.Dense(10, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

# Predicting from test data
y_pred = model.predict(X_test)
# Making binary
y_pred = np.argmax(y_pred, axis=1)
# Getting actual data
y_true = np.argmax(y_test, axis=1)

# Plotting Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Plotting training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()


print(model.predict(X_test[:4]))
print(y_test[:4])
