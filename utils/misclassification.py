import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms


def show_misclassified_images(model, test_loader, config):
    """
    Displays the misclassified images along with their corresponding targets and predictions.

    Args:
      model (torch.nn.Module): The trained model.
      test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
      config (dict): A dictionary containing configuration parameters.

    Returns:
      misclass_imgs (list): A list of misclassified images.
      misclass_targets (list): A list of misclassified targets.
      misclass_preds (list): A list of misclassified predictions.
    """
    model.eval()
    misclass_data = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config["device"]), labels.to(config["device"])

            for image, label in zip(images, labels):
                image = image.unsqueeze(0)
                outputs = model(image)
                pred = outputs.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log probability

                if pred.item() != label.item():
                    misclass_data.append((image, label, pred))

    return misclass_data


def plt_misclassified_images(config, misclass_data, max_images=10):
    """
    Plot misclassified images along with their predicted and actual labels.

    Args:
      config (dict): A dictionary containing configuration settings.
      misclass_imgs (list): A list of misclassified images.
      misclass_targets (list): A list of misclassified targets (actual labels).
      misclass_preds (list): A list of misclassified predictions.
      max_images (int, optional): The maximum number of images to plot. Defaults to 10.
    """

    def inv_normalize(image):
        inv_normalize = transforms.Normalize(
            mean=[-0.50 / 0.23, -0.50 / 0.23, -0.50 / 0.23],
            std=[1 / 0.23, 1 / 0.23, 1 / 0.23],
        )
        image = inv_normalize(image)
        image = image.squeeze(0)
        image = image.permute(1, 2, 0)
        return image

    # Determine the number of images to plot (max 10)
    n_images = max_images
    classes = config["classes"]
    fig = plt.figure(figsize=(20, 4))
    for i in range(n_images):
        misclass_imgs, misclass_targets, misclass_preds = misclass_data[i - 1]
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        # Normalize the image
        im_ = inv_normalize(misclass_imgs)
        im_ = im_.cpu().numpy()
        pred = misclass_preds.item()
        label = misclass_targets.item()

        ax.imshow(im_)
        ax.set_title(f"Prediction: {classes[pred]}\nActual: {classes[label]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
