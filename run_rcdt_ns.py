import json
from sklearn.metrics import accuracy_score
import numpy as np
import time
from pytranskit.classification.utils import load_data, take_train_samples
from pytranskit.classification.rcdt_ns import RCDT_NS

def main():
    print("Running RCDT-NS...")
    start_time = time.time()

    
    dataset = 'MNIST'
    datadir = './data'
    num_classes = 10
    n_samples_perclass = 512

    
    (x_train, y_train), (x_test, y_test) = load_data(dataset, num_classes, datadir)

    
    x_train_sub, y_train_sub = take_train_samples(
        x_train, y_train, n_samples_perclass, num_classes, repeat=0
    )


    theta = np.linspace(0, 176, 45)
    rcdt_ns_obj = RCDT_NS(num_classes, theta, rm_edge=True)
    rcdt_ns_obj.fit(x_train_sub, y_train_sub)

    
    preds = rcdt_ns_obj.predict(x_test)
    accuracy = accuracy_score(y_test, preds)
    elapsed_time = time.time() - start_time

    
    results = {
        "method": "RCDT-NS",
        "accuracy": accuracy,
        "time": elapsed_time
    }
    with open("rcdt_results.json", "w") as f:
        json.dump(results, f)

    print(f"RCDT-NS Results: {results}")

if __name__ == "__main__":
    main()
