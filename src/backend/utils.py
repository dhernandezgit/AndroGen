import os
import json
import cv2
import gradio as gr
import numpy as np
from PIL import Image
import threading

# Create a semaphore for controlling access to JSON file operations.
json_semaphore = threading.Semaphore()

def save_gif(input_folder, n_seq:int,save_path="resources/temp.gif", duration=100, loop=0):
    image_list = [file for file in os.listdir(input_folder) if file.startswith(f"{n_seq:06d}")]
    image_list = sorted(image_list, key=lambda x: int(x.split(".")[0]))
    # Open images and append them to a list
    frames = [Image.open(os.path.join(input_folder, img)) for img in image_list]
    
    # Save as an animated GIF
    frames[0].save(
        os.path.join(os.getcwd(), save_path),
        save_all=True,
        append_images=frames[1:],  # Add the rest of the frames
        duration=duration,  # Duration of each frame in milliseconds
        loop=loop  # Loop forever (set loop=1 for no loop)
    )
    

def read_json(path):
    with json_semaphore:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    
def write_json(path, data):
    with json_semaphore:
        with open(path, 'w') as file:
            json.dump(data, file, indent=4)

def update_json(path, updates):
    """
    Reads a JSON file, updates only the specified keys with new values, and writes it back.

    :param file_path: Path to the JSON file.
    :param updates: Dictionary containing the updated values.
    """
    with json_semaphore:
        # Read the existing JSON data from the file
        with open(path, 'r') as file:
            data = json.load(file)

        # Update the JSON data with the values in the updates dictionary
        for key, value in updates.items():
            if key in data:
                data[key] = value
            else:
                print(f"Key '{key}' not found in JSON file. Skipping update for this key.")

        # Write the updated JSON data back to the file
        with open(path, 'w') as file:
            json.dump(data, file, indent=4)


# Function to extract the color of a bounding box
def get_color(annotation, threshold=10):
    if annotation["boxes"]:
        box = annotation["boxes"][0]
        crop = annotation["image"][
            box["ymin"]:box["ymax"],
            box["xmin"]:box["xmax"]
        ]
        
        # Calculate the mean color
        mean_color = np.mean(crop, axis=(0, 1))
        
        # Get the central pixel's color
        center_y = (box["ymin"] + box["ymax"]) // 2
        center_x = (box["xmin"] + box["xmax"]) // 2
        central_color = annotation["image"][center_y, center_x]
        
        # Calculate Euclidean distance between mean and central color
        distance = np.linalg.norm(mean_color - central_color)
        
        # Use central color if far from mean color
        if distance > threshold:
            color = central_color
        else:
            color = mean_color
        
        return f"rgba({int(color[0])}, {int(color[1])}, {int(color[2])}, 1)"
    else:
        return "rgba(0, 0, 0, 1)"