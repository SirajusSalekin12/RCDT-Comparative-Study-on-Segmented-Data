from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.io as sio
from pytranskit.classification.utils import *
from pytranskit.classification.rcdt_ns import RCDT_NS

def main():

    mat_file_path = './fashion-mnist-master/data/fashion/converted_data.mat'

    mat_data = sio.loadmat(mat_file_path)


    x_data = mat_data['xxO']  # Use 'xxO' as the key for the data
    y_data = mat_data['label'].flatten()  # Use 'label' as the key for labels


    x_data = x_data / 255.0


    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )


    n_samples_perclass = 200
    num_classes = len(np.unique(y_train))
    x_train_sub, y_train_sub = take_train_samples(
        x_train, y_train, n_samples_perclass, num_classes, repeat=0
    )


    theta = np.linspace(0, 176, 45)
    rcdt_ns_obj = RCDT_NS(num_classes, theta, rm_edge=True)


    rcdt_ns_obj.fit(x_train_sub, y_train_sub)


    use_gpu = True
    preds = rcdt_ns_obj.predict(x_test, use_gpu)


    print('\nTest accuracy: {}%'.format(100 * accuracy_score(y_test, preds)))

if __name__ == '__main__':
    main()
