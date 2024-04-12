import math
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms

def display_gradcam_output(
    misclass_data,
    classes,
    model,
    target_layers,
    no_samples: int = 10,
    transparence: float = 0.60,
):
    """
    Display the GradCAM output for a given set of misclassified images.

    Args:
        misclass_data (list): List of misclassified images and their corresponding labels.
        classes (list): List of class labels.
        model (torch.nn.Module): The model for which GradCAM needs to be visualized.
        target_layers (str or list): The target layer(s) for GradCAM.
        no_samples (int, optional): Number of samples to display. Defaults to 10.
        transparence (float, optional): Transparency level for the GradCAM overlay. Defaults to 0.60.
    """

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    # Denormalize the data using test mean and std deviation
    inv_normalize = transforms.Normalize(
        mean=[-0.50/0.23, -0.50/0.23, -0.50/0.23],
        std=[1/0.23, 1/0.23, 1/0.23]
    ) 

    # Plot configuration
    fig = plt.figure(figsize=(10, 10))
    x_count = 5
    if no_samples is None:
        y_count = 1
    else:
        y_count = math.floor(no_samples / x_count)

    # Create an object for GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # Iterate over number of specified images
    for i in range(no_samples):
        plt.subplot(y_count, x_count, i + 1)
        input_tensor = misclass_data[i][0]

        # Get the activations of the layer for the images
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]

        # Get the original image
        img = input_tensor.squeeze(0).to("cpu")
        img = inv_normalize(img)
        rgb_img = np.transpose(img.cpu().numpy(), [1, 2, 0])

        # Mix the normal image with the activations
        visualization = show_cam_on_image(
            rgb_img, grayscale_cam, use_rgb=True, image_weight=transparence
        )

        # Plot the GradCAM output along with the original image
        plt.imshow(visualization)
        plt.title(
            f"Prediction: {classes[misclass_data[i][2].item()]}\n Actual: {classes[misclass_data[i][1].item()]}"
        )
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()
