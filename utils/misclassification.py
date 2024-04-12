"""
Module for displaying misclassified images.
"""

import math

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


def plt_misclassified_images(config, misclass_data, no_samples=10):
  """
  Plot misclassified images along with their predicted and actual labels.

  Args:
    config (dict): Configuration dictionary containing classes information.
    misclass_data (list): List of misclassified images, targets, and predictions.
    no_samples (int, optional): Number of misclassified images to plot. Defaults to 10.
  """

  x_count = 5
  if no_samples is None:
    y_count = 1
  else:
    y_count = math.floor(no_samples / x_count)

  def inv_normalize(image):
    inv_normalize = transforms.Normalize(
      mean=[-0.50 / 0.23, -0.50 / 0.23, -0.50 / 0.23],
      std=[1 / 0.23, 1 / 0.23, 1 / 0.23],
    )
    image = inv_normalize(image)
    image = image.squeeze(0)
    image = image.permute(1, 2, 0)
    return image

  # Determine the number of images to plot
  classes = config["classes"]
  fig = plt.figure(figsize=(20, 4))
  for i in range(no_samples):
    misclass_imgs, misclass_targets, misclass_preds = misclass_data[i - 1]
    plt.subplot(y_count, x_count, i + 1)
    # Normalize the image
    im_ = inv_normalize(misclass_imgs)
    im_ = im_.cpu().numpy()
    pred = misclass_preds.item()
    label = misclass_targets.item()

    plt.imshow(im_)
    plt.title(f"Prediction: {classes[pred]}\nActual: {classes[label]}")
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])

  plt.tight_layout()
  plt.show()
