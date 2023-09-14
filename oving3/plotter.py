import torch
import matplotlib.pyplot as plt

def plot_test_images(task, x_test, y_test, model):
    #model.eval()
    plt.figure(figsize=(10, 10))
    random_indices = torch.randint(0, len(x_test), (25,))  # Generate 25 random indices

    for i, idx in enumerate(random_indices):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[idx].cpu().reshape(28, 28), cmap=plt.cm.binary)
        predicted_label = model.f(x_test[idx:idx + 1]).argmax(1).item()
        true_label = y_test[idx].argmax().item()
        plt.xlabel(f"Predicted: {predicted_label}, True: {true_label}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(task, fontsize=16)
    plt.show()
    #model.train()