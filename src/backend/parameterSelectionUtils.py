import os
import sys
sys.path.append(os.path.abspath('.'))

import json
import gradio as gr

from src.backend.utils import read_json

# Global variable
species_path = ""
species_dict = {}

# Get method
def get_species_path():
    """
    Returns the value of the global variable species_path.
    """
    global species_path
    return species_path

# Set method
def set_species_path(path):
    """
    Sets the value of the global variable species_path.
    Parameters:
        path (str): The new path to set for species_path.
    """
    global species_path
    species_path = path
    
# Get method
def get_species_dict():
    """
    Returns the value of the global variable species_dict.
    """
    global species_dict
    return species_dict

# Set method
def set_species_dict(new_dict):
    """
    Sets the value of the global variable species_dict.
    Parameters:
        new_dict (dict): The new dictionary to set for species_dict.
    """
    global species_dict
    if isinstance(new_dict, dict):
        species_dict = new_dict
    else:
        raise TypeError("species_dict must be a dictionary.")
    
def save_custom_json(json_text, name):
    with open(os.path.join(get_species_path(), f"{name}.json"), "w", encoding="utf-8") as file:
        json.dump(json.loads(json_text), file, indent=4)
    return gr.update(choices=species_update_list())

def species_update_list():
    return [s[:-5] for s in os.listdir(get_species_path())] + ["Custom"]

def toggle_components(selection, custom_json):
    if selection == "Custom":
        return [
    gr.update(choices=species_update_list()),
    gr.update(choices=list(get_species_dict().keys()), value=list(get_species_dict().keys())[0]),
    gr.update(visible=True, value=json.dumps(get_species_dict(), indent=4)),
    gr.update(visible=False),
    gr.update(visible=True), 
    gr.update(visible=True)
    ]
    else:
        set_species_dict(read_json(os.path.join(species_path, f"{selection}.json")))
        return [
    gr.update(choices=species_update_list()),
    gr.update(choices=list(get_species_dict().keys()), value=list(get_species_dict().keys())[0]),
    gr.update(visible=False),
    gr.update(visible=True, value=get_species_dict()),
    gr.update(visible=False), 
    gr.update(visible=False)
    ]