import numpy as np
from sklearn.metrics import accuracy_score
from PIL import Image
import pickle
import os
import sys

sys.path.append('../')
from pytranskit.classification.utils import take_train_samples
from pytranskit.classification.rcdt_ns import RCDT_NS
from scipy.io import loadmat


def load_cifar10_data(data_dir, num_classes=10, target_size=(24, 24)):
    x_train, y_train = [], []
    for i in range(1, 6):
        data_file = os.path.join(data_dir, f"data_batch_{i}.mat")
        batch = loadmat(data_file)
        x_train.append(batch['data'])
        y_train.extend(batch['labels'].flatten())

    x_train = np.vstack(x_train).reshape(-1, 3, 32, 32).astype("float32")
    y_train = np.array(y_train)


    test_file = os.path.join(data_dir, "test_batch.mat")
    test_batch = loadmat(test_file)
    x_test = test_batch['data'].reshape(-1, 3, 32, 32).astype("float32")
    y_test = test_batch['labels'].flatten()


    x_train = np.transpose(x_train, (0, 2, 3, 1))  # Convert to HWC
    x_test = np.transpose(x_test, (0, 2, 3, 1))


    x_train = np.array([Image.fromarray(img.astype(np.uint8)).convert('L').resize(target_size) for img in x_train])
    x_test = np.array([Image.fromarray(img.astype(np.uint8)).convert('L').resize(target_size) for img in x_test])

    return (x_train, y_train), (x_test, y_test)


def main():

    datadir = './cifar'

    num_classes = 10  # CIFAR-10 has 10 classes


    (x_train, y_train), (x_test, y_test) = load_cifar10_data(datadir, num_classes)


    n_samples_perclass = 100
    x_train_sub, y_train_sub = take_train_samples(
        x_train, y_train, n_samples_perclass, num_classes, repeat=0
    )

    # Create an instance of RCDT_NS class
    theta = np.linspace(0, 176, 45)
    rcdt_ns_obj = RCDT_NS(num_classes, theta, rm_edge=True)


    rcdt_ns_obj.fit(x_train_sub, y_train_sub)


    use_gpu = True
    preds = rcdt_ns_obj.predict(x_test, use_gpu)


    print('\nTest accuracy: {}%'.format(100 * accuracy_score(y_test, preds)))


if __name__ == '__main__':
    main()
