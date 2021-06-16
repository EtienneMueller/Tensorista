import tensorslow as tf


tf.random.set_seed(1234)

# Load in the data
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
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,), name='dense1'),
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
