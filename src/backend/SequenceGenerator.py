import os
import sys
sys.path.append(os.path.abspath('.'))

import json
import math
import numpy as np
import tqdm
import cv2
import configparser
import threading
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from itertools import islice

from src.backend.SpermatozoonFactory import SpermatozoonFactory
from src.backend.DebrisFactory import DebrisFactory
from src.backend.BackgroundGenerator import BackgroundGenerator
from src.backend.ConfigParser import SequenceConfig


def generate_frame_wrapper(args):
    # Unpack arguments and call the original method
    obj, save_path, file_name, frame_index, elements = args
    obj.generate_frame(save_path, file_name, frame_index, elements)


# Spermatozoon and its Morphology
class SequenceGenerator:
    def __init__(self, num_frames: int, sequence_config: SequenceConfig, sf = SpermatozoonFactory, df = DebrisFactory, bgg=BackgroundGenerator, dpi=300, image_augmentor=None):
        self.id = 0
        self.num_frames = num_frames
        self.sequence_config = sequence_config
        self.sequence_config.update() 
        
        self.params = self.sequence_config.getParameters()
        
        self.sf = sf
        self.df = df
        self.bgg = bgg
        self.image_augmentor = image_augmentor
        
        self.dpi = dpi
        
    def _generate_figure(self):
        width, height = self.params["Sequence.resolution"]["resolution"]
        
        # Calculate figure size in inches based on desired resolution and dpi
        fig_width = width / self.dpi
        fig_height = height / self.dpi
        
        # Create a figure with the calculated size
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=self.dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # Remove padding by placing axis to fill the figure
        
        # Remove borders, axes, and ticks
        ax.axis('off')

        # Set axis limits to match the image size in pixels
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        return fig, ax

    def _save_figure(self, fig, path, name):
        save_path = os.path.join(path, name)
        # Save the figure without borders
        fig.savefig(save_path, dpi=self.dpi, pad_inches=0)
        plt.close(fig)  # Close the figure to free resources
        
    def _generate_elements(self):
        resolution = self.params["Sequence.resolution"]["resolution"]
        background=self.bgg.getBackground(resolution=resolution)
        spermatozoa = []
        debrises = []
        for n in range(self.params["Sequence.quantities"]["spermatozoon_n"]):
            spermatozoa.append(self.sf.getSample(resolution=resolution, total_frames=self.num_frames))
        for n in range(self.params["Sequence.quantities"]["debris_n"]):
            debrises.append(self.df.getSample(resolution=resolution, total_frames=self.num_frames))
        return (background, spermatozoa, debrises)
    
    def _pixel_diameter_to_scatter_size(self, diameter_in_pixels):
        """
        Converts a diameter in pixels to the size required by ax.scatter in Matplotlib.

        Parameters:
            diameter_in_pixels (float): The diameter of the marker in pixels.
            dpi (float): The dots per inch (DPI) of the Matplotlib figure.

        Returns:
            float: The marker size for ax.scatter.
        """
        # Convert diameter in pixels to points (1 inch = 72 points)
        diameter_in_points = diameter_in_pixels * 144.0 / self.dpi
        # Scatter size is proportional to the area (diameter^2)
        sizes = (diameter_in_points / 2) ** 2
        return sizes

    def _render(self, ax, xs, ys, sizes, rgba_colors):
        fixed_sizes = self._pixel_diameter_to_scatter_size(sizes)
        # Create a scatter plot for the circles with RGBA colors
        scatter = ax.scatter(xs, ys, s=fixed_sizes, marker='o', c=rgba_colors, edgecolors='none', linewidth=0)

    def generate_frame(self, save_path: str, name=None, t=0, elements=None):
        os.makedirs(save_path, exist_ok=True)
        if elements is None:
            background, spermatozoa, debrises = self._generate_elements()
        else:
            background, spermatozoa, debrises = elements
        if name is None:
            name = f"{t:06d}.png"
            
        fig, ax = self._generate_figure()
        ax.imshow(background)
        
        element_xs, element_ys, element_sizes, element_rgba_colors = np.array([]), np.array([]), np.array([]), np.empty((0,4))
        shadow_xs, shadow_ys, shadow_sizes, shadow_rgba_colors = np.array([]), np.array([]), np.array([]), np.empty((0,4))
        highlight_xs, highlight_ys, highlight_sizes, highlight_rgba_colors = np.array([]), np.array([]), np.array([]), np.empty((0,4))
        full_xs, full_ys, full_sizes, full_rgba_colors = np.array([]), np.array([]), np.array([]), np.empty((0,4))
        for s in spermatozoa:
            # Generate and render the spermatozoon
            xs, ys, sizes, rgba_colors = s.calculate_movement(t=t)
            element_xs = np.concatenate((element_xs, xs))
            element_ys = np.concatenate((element_ys, ys))
            element_sizes = np.concatenate((element_sizes, sizes))
            element_rgba_colors = np.concatenate((element_rgba_colors, rgba_colors))
            
            sxs, sys, ssizes, srgba_colors = s.add_shadows(xs, ys, sizes, rgba_colors)
            shadow_xs= np.concatenate((shadow_xs, sxs))
            shadow_ys = np.concatenate((shadow_ys, sys))
            shadow_sizes = np.concatenate((shadow_sizes, ssizes))
            shadow_rgba_colors = np.concatenate((shadow_rgba_colors, srgba_colors))
            
            hxs, hys, hsizes, hrgba_colors = s.add_highlight(xs, ys, sizes, rgba_colors)
            highlight_xs = np.concatenate((highlight_xs, hxs))
            highlight_ys = np.concatenate((highlight_ys, hys))
            highlight_sizes = np.concatenate((highlight_sizes, hsizes))
            highlight_rgba_colors = np.concatenate((highlight_rgba_colors, hrgba_colors))
            
        for d in debrises:
            # Generate and render the spermatozoon
            xs, ys, sizes, rgba_colors = d.calculate_movement(t=t)
            element_xs = np.concatenate((element_xs, xs))
            element_ys = np.concatenate((element_ys, ys))
            element_sizes = np.concatenate((element_sizes, sizes))
            element_rgba_colors = np.concatenate((element_rgba_colors, rgba_colors))
            
            xs, ys, sizes, rgba_colors = d.add_shadows(xs, ys, sizes, rgba_colors)
            shadow_xs = np.concatenate((shadow_xs, xs))
            shadow_ys = np.concatenate((shadow_ys, ys))
            shadow_sizes = np.concatenate((shadow_sizes, sizes))
            shadow_rgba_colors = np.concatenate((shadow_rgba_colors, rgba_colors))
            
        full_xs = np.concatenate((shadow_xs, element_xs, highlight_xs))
        full_ys = np.concatenate((shadow_ys, element_ys, highlight_ys))
        full_sizes = np.concatenate((shadow_sizes, element_sizes, highlight_sizes))        
        full_rgba_colors = np.concatenate((shadow_rgba_colors, element_rgba_colors, highlight_rgba_colors))
            
        self._render(ax, full_xs, full_ys, full_sizes, full_rgba_colors)
        self._save_figure(fig, save_path, name)
        if self.image_augmentor is not None:
            im = cv2.imread(os.path.join(save_path, name))
            cv2.imwrite(os.path.join(save_path, name), self.image_augmentor.augment(im.copy(), num_images=1)[0])    
            
            
    def generate_sequence(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.sequence_config.update()
        self.params = self.sequence_config.getParameters()
        
        elements = self._generate_elements()
        background, spermatozoa, debrises = elements
        
        # Launch threads in a loop
        #for t in tqdm.tqdm(range(self.num_frames)):
        #    self.generate_frame(save_path, f"{t:06d}.png", t, elements)
            
        tasks = [
            (self, output_dir, f"{self.id:06d}_{t:06d}.png", t, elements)
            for t in range(self.num_frames)
        ]
        
        # Use multiprocessing Pool with batching
        num_workers = min(cpu_count(), len(tasks))  # Use the lesser of CPU count or task size
        batch_size = max(1, len(tasks) // (10 * num_workers))  # Adjust batch size for efficiency
        
        with Pool(processes=num_workers) as pool:
            # Process tasks in chunks for better resource utilization
            for _ in tqdm.tqdm(
                pool.imap_unordered(generate_frame_wrapper, tasks, chunksize=batch_size),
                total=self.num_frames,
                desc="Rendering frames"
            ):
                pass
        self.id += 1
        
            
if __name__ == "__main__":
    config_path = os.path.join(os.getcwd(), 'cfg', 'config.ini')
    sequence_config = SequenceConfig(config_path)
    
    sf = SpermatozoonFactory(species_dict_path="cfg/species/sampleSpecie.json", style_config_path="cfg/styles/base.json")
    df = DebrisFactory(debris_dict_path="cfg/debris.json", style_config_path="cfg/styles/base.json")
    
    paths = "/home/daniel/Documents/Projects/Kubus/Morfología/Data/vids_processed"
    image_paths = os.listdir(paths)
    image_paths = [os.path.join(paths, p) for p in image_paths]
    bgg = BackgroundGenerator()
    bgg.setGenerationMethod('list', paths=image_paths)


    sg = SequenceGenerator(25, sequence_config, sf, df, bgg)
    sg.generate_sequence("results_new")