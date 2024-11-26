import json
import matplotlib.pyplot as plt

def load_results(filename):
    with open(filename, "r") as f:
        return json.load(f)

def plot_comparison(rcdt_results, cnn_results):
    """Plot comparison of accuracy and runtime."""
    methods = [rcdt_results['method'], cnn_results['method']]
    accuracies = [rcdt_results['accuracy'] * 100, cnn_results['accuracy'] * 100]
    runtimes = [rcdt_results['time'], cnn_results['time']]

    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    
    ax[0].bar(methods, accuracies, color=['blue', 'orange'])
    ax[0].set_title('Accuracy Comparison')
    ax[0].set_ylabel('Accuracy (%)')
    for i, v in enumerate(accuracies):
        ax[0].text(i, v + 1, f"{v:.2f}%", ha='center')

    
    ax[1].bar(methods, runtimes, color=['blue', 'orange'])
    ax[1].set_title('Runtime Comparison')
    ax[1].set_ylabel('Time (seconds)')
    for i, v in enumerate(runtimes):
        ax[1].text(i, v + 0.5, f"{v:.2f}s", ha='center')

    plt.tight_layout()
    plt.show()

def main():
    
    rcdt_results = load_results("rcdt_results.json")
    cnn_results = load_results("cnn_results.json")

    
    print("RCDT-NS Results:", rcdt_results)
    print("CNN Results:", cnn_results)


    plot_comparison(rcdt_results, cnn_results)

if __name__ == "__main__":
    main()
