import json
import os
import matplotlib.pyplot as plt

def find_metrics_files(base_path):
    metrics_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == 'metrics.json':
                full_path = os.path.join(root, file)
                metrics_files.append(full_path)
    return metrics_files

def read_json_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            try:
                entry = json.loads(line.strip())
                # Ensure 'total_loss' is in the entry to consider it
                if 'total_loss' in entry and 'fast_rcnn/false_negative' in entry:
                    data.append(entry)
            except json.JSONDecodeError:
                continue
    return data

def plot_data(data, filepath):
    # Extract data
    iterations = [item['iteration'] for item in data]
    total_losses = [item['total_loss'] for item in data]
    loss_box_reg = [item['loss_box_reg'] for item in data]
    loss_cls = [item['loss_cls'] for item in data]
    learning_rates = [item['lr'] for item in data]
    for i, item in enumerate(data):
        try:
            item['fast_rcnn/false_negative']
        except:
            print(i)
            print(item)

        
    false_negative = [item['fast_rcnn/false_negative'] for item in data]
    cls_accuracies = [item['fast_rcnn/cls_accuracy'] for item in data]
    fg_cls_accuracies = [item['fast_rcnn/fg_cls_accuracy'] for item in data]
    
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 15))

    # Total loss plot
    axs[0].plot(iterations, total_losses, label='Total Loss', color='red')
    axs[0].set_title('Total Loss over Iterations')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Total Loss')
    axs[0].grid(True)

    # Plot Total Loss
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, total_losses, label='Total Loss', color='red')
    plt.plot(iterations, loss_box_reg, label='Box Regression Loss', color='blue')
    plt.plot(iterations, loss_cls, label='Classification Loss', color='green')
    plt.title('Total Loss over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Total Loss')
    plt.grid(True)
    output_path = os.path.join(os.path.dirname(filepath), 'plot_results_loss.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

    # Classification accuracy plot
    axs[1].plot(iterations, cls_accuracies, label='Fast R-CNN Class Accuracy', color='green')
    axs[1].plot(iterations, fg_cls_accuracies, label='Fast R-CNN FG Class Accuracy', color='purple')
    axs[1].plot(iterations, false_negative, label='Fast R-CNN False Negative', color='red')
    axs[1].set_title('Classification Accuracy over Iterations')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    # Plot Classification Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, cls_accuracies, label='Fast R-CNN Class Accuracy', color='green')
    plt.plot(iterations, fg_cls_accuracies, label='Fast R-CNN FG Class Accuracy', color='purple')
    plt.plot(iterations, false_negative, label='Fast R-CNN False Negative', color='red')
    plt.title('Classification Accuracy over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(os.path.dirname(filepath), 'plot_results_cls.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

    # Save plot to the same directory as the metrics.json file
    output_path = os.path.join(os.path.dirname(filepath), 'plot_results_all.png')
    plt.savefig(output_path)
    plt.close(fig)  # Close the plot to free up memory
    print(f"Plot saved to {output_path}")

def choose_file(files):
    print("Available files:")
    for index, file in enumerate(files):
        print(f"{index + 1}: {file}")
    selection = int(input("Enter the number of the file to plot: ")) - 1
    return files[selection]

def main():
    # base_path = input("Enter the base path to the outputs directory: ")
    base_path = "./outputs"
    metrics_files = find_metrics_files(base_path)
    if not metrics_files:
        print("No metrics.json files found in the directory.")
        return
    selected_file = choose_file(metrics_files)
    data = read_json_data(selected_file)
    plot_data(data, selected_file)

if __name__ == "__main__":
    main()
