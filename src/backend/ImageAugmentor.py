import albumentations as A
from albumentations.core.composition import OneOf
import cv2

class ImageAugmentor:
    def __init__(self, contrast=0.1, brightness=0.1, horizontal_flip=False, vertical_flip=False):
        """
        Initializes the ImageAugmentor class with specified augmentation parameters.

        :param contrast: Contrast adjustment factor (value >1 increases contrast; value 0-1 decreases contrast).
        :param brightness: Brightness adjustment factor (value to be added to pixel intensities).
        :param horizontal_flip: Boolean indicating whether to apply a horizontal flip.
        :param vertical_flip: Boolean indicating whether to apply a vertical flip.
        """
        self.contrast = contrast
        self.brightness = brightness
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.augmenter = self.create_augmenter()

    def create_augmenter(self):
        """
        Creates the albumentations augmenter based on the configured parameters.

        :return: An albumentations.Compose object.
        """
        augmenters = []

        # Brightness and Contrast adjustment
        augmenters.append(A.RandomBrightnessContrast(
            brightness_limit=[-float(self.brightness), float(self.brightness)],
            contrast_limit=[-float(self.contrast), float(self.contrast)],
            p=1.0
        ))

        # Horizontal flip
        if self.horizontal_flip:
            augmenters.append(A.HorizontalFlip(p=0.5))  # Horizontal flip with 50% probability

        # Vertical flip
        if self.vertical_flip:
            augmenters.append(A.VerticalFlip(p=0.5))  # Vertical flip with 50% probability

        return A.ReplayCompose(augmenters)

    def augment(self, image, num_images=1):
        """
        Applies the configured augmentations to generate multiple augmented images.

        :param image: Input image (numpy array).
        :param num_images: Number of augmented images to generate.
        :return: List of augmented images.
        """
        augmented_images = [self.augmenter(image=image) for _ in range(num_images)]
        images = []
        horizontal_flips = []
        vertical_flips = []
        for augmented_image in augmented_images:
            transform_names = [aug["__class_fullname__"] for aug in augmented_image["replay"]["transforms"]]
            horizontal_flip_index = transform_names.index("HorizontalFlip")
            vertical_flip_index = transform_names.index("VerticalFlip")
            images.append(augmented_image["image"])
            horizontal_flips.append(augmented_image["replay"]["transforms"][0]["applied"])
            vertical_flips.append(augmented_image["replay"]["transforms"][1]["applied"])
        return images, horizontal_flips, vertical_flips
