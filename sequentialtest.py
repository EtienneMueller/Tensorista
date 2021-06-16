import tensorslow as tf
# import tensorflow as tf


tf.random.set_seed(1234)

# Load in the data
# X = np.array([[0,0], [0,1], [1,0], [1,1]]) #(4,2)
# Y = np.array([[0, 1, 1, 0]]).T #(4,1)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Normalize the images:
x_train, x_test = x_train / 255., x_test / 255.

# Flatten the images.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

x_val = x_train[-10000:, :]
y_val = y_train[-10000:, ]
x_train = x_train[:-10000, :]
y_train = y_train[:-10000]

# Build the model.
model = tf.keras.models.Sequential([
        # tf.keras.layers.Input(shape=(784,)),
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,), name='dense1'),
        # tf.keras.layers.Dense(64, activation='relu', name='dense12'),
        # tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax', name='dense2'),
    ], name='seq1'
)

model.summary()

# Compile the model.
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Train the model.
model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=1000,
    validation_data=(x_val, y_val),
)

model.evaluate(x_train, y_train, batch_size=128)

"""for layer in model.layers:
    print(layer.get_weights()[0].shape)
    print(layer.get_weights()[1].shape)

print("Evaluate on test data")
results = model.evaluate(x_train, y_train, batch_size=128)
print("test loss, test acc:", results)"""

"""# Load in the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255., x_test / 255.
print("x_train.shape:", x_train.shape)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
start_time = time.time()
r = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=10
)
print("--- %s seconds ---" % (time.time() - start_time))"""
