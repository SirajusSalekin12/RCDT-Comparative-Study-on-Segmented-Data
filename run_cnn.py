import json
import tensorflow as tf
from tensorflow.keras import layers, models
import time
from pytranskit.classification.utils import load_data

def main():
    print("Running CNN...")
    start_time = time.time()

    
    dataset = 'MNIST'
    datadir = './data'
    num_classes = 10
    input_shape = (28, 28, 1)

    
    (x_train, y_train), (x_test, y_test) = load_data(dataset, num_classes, datadir)

    
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, *input_shape)
    x_test = x_test.reshape(-1, *input_shape)

    
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    elapsed_time = time.time() - start_time

    
    results = {
        "method": "CNN",
        "accuracy": test_accuracy,
        "time": elapsed_time
    }
    with open("cnn_results.json", "w") as f:
        json.dump(results, f)

    print(f"CNN Results: {results}")

if __name__ == "__main__":
    main()
