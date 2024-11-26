from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.io as sio
from pytranskit.classification.utils import *
from pytranskit.classification.rcdt_ns import RCDT_NS
from PIL import Image  # Pillow for image processing


def main():

    mat_file_path = r"E:\nsu\semester 9\cse499a\bangla_data.mat"

    mat_data = sio.loadmat(mat_file_path)

    x_data = mat_data['images']
    y_data = mat_data['labels'].flatten()


    if len(x_data.shape) == 4:  # (num_images, height, width, num_channels)

        x_data = np.array([np.array(Image.fromarray(img).convert('L')) for img in x_data])


    x_data = np.expand_dims(x_data, axis=-1)  # Shape becomes (num_images, height, width, 1)


    x_data = x_data / 255.0  # Scale pixel values to [0, 1]


    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )


    n_samples_perclass = 10
    num_classes = len(np.unique(y_train))
    x_train_sub, y_train_sub = take_train_samples(
        x_train, y_train, n_samples_perclass, num_classes, repeat=0
    )


    theta = np.linspace(0, 176, 45)
    rcdt_ns_obj = RCDT_NS(num_classes, theta, rm_edge=True)


    rcdt_ns_obj.fit(x_train_sub, y_train_sub)


    use_gpu = True
    preds = rcdt_ns_obj.predict(x_test)


    print('\nTest accuracy: {}%'.format(100 * accuracy_score(y_test, preds)))


if __name__ == '__main__':
    main()
