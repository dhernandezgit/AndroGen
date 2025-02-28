import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import math
import numpy as np
import tqdm

from src.backend.common.Pose import Pose
from src.backend.common.Color import Color
from src.backend.common.Distributions import NormalDistribution, UniformDistribution
from src.backend.Motion import *
from src.backend.Spermatozoon import *
from src.backend.utils import read_json


# Debris
class DebrisFactory:
    def __init__(self, debris_dict_path: str, style_config_path: str):
        self.debris_dict = self.loadDict(debris_dict_path)
        self.style_dict = self.loadStyle(style_config_path)
        self.id_count = 0
    
    def loadDict(self, path: str):
        """
        Load debris from a file.
        :param path: Path to the file containing the debris data
        """
        # Load JSON file into dictionary
        return read_json(path)
    
    
    def loadStyle(self, path: str):
        """
        Load morphologies for different species from a file.
        :param path: Path to the file containing the morphologies
        """
        # Load JSON file into dictionary
        return read_json(path)
    
    
    def um2pix(self, species_dict, scale = 2.35):
        for size, attributes in debris_dict.items():
            if "measurements" in attributes:
                measurements = attributes["measurements"]
                for feature, dims in measurements.items():
                    for dim, values in dims.items():
                        if isinstance(values, dict) and "mean" in values and "std" in values:
                            # Apply the scale to both mean and std
                            values["mean"] *= scale
                            values["std"] *= scale
        return debris_dict
        
    
    def getSample(self, resolution=None, total_frames=1000):
        """
        Get a sample of .
        :return: A sample of the spermatozoon's morphology
        """
        size_probs = [self.debris_dict[size]["probability"] for size in self.debris_dict.keys()]
        size = np.random.choice(list(self.debris_dict.keys()), 1, p=size_probs)[0]

        scale = self.style_dict["debris_scale"]
        pose = Pose(
            x=UniformDistribution(0, resolution[0]-1).random_samples(),
            y=UniformDistribution(0, resolution[1]-1).random_samples(),
            angle=UniformDistribution(0, 360).random_samples(),
        )
        z = UniformDistribution(a=self.style_dict["z_start"], b=self.style_dict["z_end"]).random_samples()
        max_z = max(self.style_dict["z_end"] - 1, 1 - self.style_dict["z_start"])
        blur = self.style_dict["blur_start"] + int((self.style_dict["blur_end"] - self.style_dict["blur_start"])*np.abs(1-z)/max_z)
        motion_list = np.random.choice(self.debris_dict[size]["motions"]["list"], total_frames, p=self.debris_dict[size]["motions"]["probabilities"])
        motion = Motion([globals()[item]() for item in motion_list], duration=total_frames)
        # Create Components
        debris = Head(
            width=scale*NormalDistribution(mean=self.debris_dict[size]["measurements"]["mean"], std=self.debris_dict[size]["measurements"]["std"]).random_samples(),
            height=scale*NormalDistribution(mean=self.debris_dict[size]["measurements"]["mean"], std=self.debris_dict[size]["measurements"]["std"]).random_samples(),
            scale_highlight=1,
            offset_highlight=0,
            color=Color(r=self.style_dict["color"]["debris"]["r"], g=self.style_dict["color"]["debris"]["g"], b=self.style_dict["color"]["debris"]["b"]),
            color_highlight=Color(r=self.style_dict["color"]["debris"]["r"], g=self.style_dict["color"]["debris"]["g"], b=self.style_dict["color"]["debris"]["b"])
        )
            
        shadow = Shadow(
            starting_color=Color(r=self.style_dict["color"]["debris_shadow_start"]["r"], g=self.style_dict["color"]["debris_shadow_start"]["g"], b=self.style_dict["color"]["debris_shadow_start"]["b"]),
            ending_color=Color(r=self.style_dict["color"]["debris_shadow_end"]["r"], g=self.style_dict["color"]["debris_shadow_end"]["g"], b=self.style_dict["color"]["debris_shadow_end"]["b"]),
            n_iterations=self.style_dict["debris_shadow_n"],
            offset=0,
            starting_scale=self.style_dict["debris_shadow_starting_scale"]/4 if size == "Large" else self.style_dict["debris_shadow_starting_scale"]/1.5,
            ending_scale=self.style_dict["debris_shadow_ending_scale"]/4 if size == "Large" else self.style_dict["debris_shadow_ending_scale"]/1.5
        )
        
        debris = Spermatozoon(
            sperm_id=self.id_count,
            morphology="Normal",
            pose=pose,
            z=z,
            motion=motion,
            components=[debris],
            shadow=shadow,
            blur=blur,
            n_points=self.style_dict["n_points"]
        )
        self.id_count += 1
        return debris
    
    
def pixel_diameter_to_scatter_size(diameter_in_pixels, dpi=300):
    """
    Converts a diameter in pixels to the size required by ax.scatter in Matplotlib.

    Parameters:
        diameter_in_pixels (float): The diameter of the marker in pixels.
        dpi (float): The dots per inch (DPI) of the Matplotlib figure.

    Returns:
        float: The marker size for ax.scatter.
    """
    # Convert diameter in pixels to points (1 inch = 72 points)
    diameter_in_points = diameter_in_pixels * 144.0 / dpi
    # Scatter size is proportional to the area (diameter^2)
    sizes = (diameter_in_points / 2) ** 2
    return sizes

def render(ax, xs, ys, sizes, rgba_colors):
    fixed_sizes = pixel_diameter_to_scatter_size(sizes)
    # Create a scatter plot for the circles with RGBA colors
    scatter = ax.scatter(xs, ys, s=fixed_sizes, marker='o', c=rgba_colors, edgecolors='none', linewidth=0)
    
    
if __name__ == "__main__":
    sf = SpermatozoonFactory(species_dict_path="cfg/species/sampleSpecie.json", style_config_path="cfg/styles/base.json")

    # Define desired resolution in pixels
    width, height = 1280, 1024

    # Calculate figure size in inches based on desired resolution and dpi
    dpi = 300  # You can adjust this DPI value as needed
    fig_width = width / dpi
    fig_height = height / dpi

    N = 100
    time = 100
    spermatozoa = []
    for n in range(N):
        spermatozoa.append(sf.getSample())
            
    for t in tqdm.tqdm(range(time)):
        # Create a figure with the calculated size
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # Remove padding by placing axis to fill the figure
        
        # Remove borders, axes, and ticks
        ax.axis('off')

        # Set axis limits to match the image size in pixels
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)

        # Invert the y-axis to match the pixel coordinate system (origin at top-left)
        #ax.invert_yaxis()
        ax.grid(visible=True, linestyle='-', color='gray', alpha=1)

        spermatozoa_xs, spermatozoa_ys, spermatozoa_sizes, spermatozoa_rgba_colors = np.array([]), np.array([]), np.array([]), np.empty((0,4))
        shadow_xs, shadow_ys, shadow_sizes, shadow_rgba_colors = np.array([]), np.array([]), np.array([]), np.empty((0,4))
        full_xs, full_ys, full_sizes, full_rgba_colors = np.array([]), np.array([]), np.array([]), np.empty((0,4))
        for s in spermatozoa:
            # Generate and render the spermatozoon
            xs, ys, sizes, rgba_colors = s.calculate_movement(t=t)
            spermatozoa_xs = np.concatenate((spermatozoa_xs, xs))
            spermatozoa_ys = np.concatenate((spermatozoa_ys, ys))
            spermatozoa_sizes = np.concatenate((spermatozoa_sizes, sizes))
            spermatozoa_rgba_colors = np.concatenate((spermatozoa_rgba_colors, rgba_colors))
            
            xs, ys, sizes, rgba_colors = s.add_shadows(xs, ys, sizes, rgba_colors)
            shadow_xs= np.concatenate((shadow_xs, xs))
            shadow_ys = np.concatenate((shadow_ys, ys))
            shadow_sizes = np.concatenate((shadow_sizes, sizes))
            shadow_rgba_colors = np.concatenate((shadow_rgba_colors, rgba_colors))
            
        full_xs = np.concatenate((shadow_xs, spermatozoa_xs))
        full_ys = np.concatenate((shadow_ys, spermatozoa_ys))
        full_sizes = np.concatenate((shadow_sizes, spermatozoa_sizes))        
        full_rgba_colors = np.concatenate((shadow_rgba_colors, spermatozoa_rgba_colors))
            
        render(ax, full_xs, full_ys, full_sizes, full_rgba_colors)

        # Save the figure without borders
        fig.savefig(f"results/output_image_{t}.png", dpi=dpi, pad_inches=0)
        plt.close(fig)  # Close the figure to free resources