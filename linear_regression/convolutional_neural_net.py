import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), input_shape=(28, 28, 1)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(units=10),
    tf.keras.layers.Softmax()
])

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)

print(model.evaluate(X_test, y_test))

# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
#
# D:\Data Core Systems\New folder\Game theory\cudnn-11.2-windows-x64-v8.1.1.33\cuda\bin
#
