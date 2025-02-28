import os
import cv2
import numpy as np
import random

class BackgroundGenerator:
    def __init__(self):
        self.generation_method = None
        self.image_paths = None
        self.single_image_path = None
    
    def setGenerationMethod(self, method, paths=None):
        """
        Set the generation method: 'list' or 'single image'.
        For 'list', provide a list of image paths.
        For 'single image', provide a single image path.
        """
        if method not in ['list', 'single']:
            raise ValueError("Invalid generation method. Use 'list' or 'single'.")
        
        self.generation_method = method
        self.image_paths = paths

    def _generateFromList(self, n_images=10):
        """
        Select a maximum of 10 randomly chosen image files from the list,
        calculate the median of these images, and return it.
        """
        if not self.image_paths or len(self.image_paths) == 0:
            raise ValueError("Image paths list is empty or not provided.")
        
        # Select a maximum of 20 random image paths
        selected_paths = random.sample(self.image_paths, min(n_images, len(self.image_paths)))
        
        ref_img = cv2.imread(selected_paths[0])
        ref_shape = ref_img.shape
        
        images = []
        for path in selected_paths:
            img = cv2.imread(path)
            if img is not None:
                if img.shape != ref_shape:
                    images.append(cv2.resize(img, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_LINEAR))
                else:
                    images.append(img)
        
        if len(images) == 0:
            raise ValueError("No valid images found in the provided paths.")
        
        # Stack images and compute the median
        stacked_images = np.stack(images, axis=3)  # Add new dimension for stacking
        median_image = np.median(stacked_images, axis=3).astype(np.uint8)  # Median across the stacked axis
        
        return median_image

    def _setFromImage(self):
        """
        Return the image given the image path.
        """
        if not self.image_paths:
            img = np.zeros((1280,1024,3), dtype=np.uint8)
        else:
            img = cv2.imread(self.image_paths)
        if img is None:
            raise ValueError("Image could not be read. Check the path.")
        
        return img

    def getBackground(self, resolution=None):
        """
        Return the background image based on the selected generation method.
        """
        if self.generation_method == 'list':
            background = self._generateFromList()
        elif self.generation_method == 'single':
            background = self._setFromImage()
        else:
            raise ValueError("Generation method is not set. Use setGenerationMethod().")
        if resolution is not None:
            width, height = resolution
            # Resize the image to the specified dimensions
            background = cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR)
    
        return background

if __name__ == "__main__":
    # Initialize the generator
    bg_generator = BackgroundGenerator()

    # Using a list of image paths
    paths = "/home/daniel/Documents/Projects/Kubus/Morfología/Data/vids_processed"
    image_paths = os.listdir(paths)
    image_paths = [os.path.join(paths, p) for p in image_paths]

    bg_generator.setGenerationMethod('list', paths=image_paths)
    background_from_list = bg_generator.getBackground()
    cv2.imwrite("background_from_list_1.jpg", background_from_list)
    background_from_list = bg_generator.getBackground()
    cv2.imwrite("background_from_list_2.jpg", background_from_list)

    # Using a single image path
    single_image_path = image_paths[0]
    bg_generator.setGenerationMethod('single', single_image_path=single_image_path)
    background_from_image = bg_generator.getBackground()
    cv2.imwrite("background_from_image.jpg", background_from_image)