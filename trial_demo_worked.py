
from sklearn.metrics import accuracy_score
import sys
sys.path.append('../')
from pytranskit.classification.utils import *
from pytranskit.classification.rcdt_ns import RCDT_NS


def main():

    datadir = './data'
    dataset = 'MNIST'
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = load_data(dataset, num_classes, datadir)


    n_samples_perclass = 512
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
