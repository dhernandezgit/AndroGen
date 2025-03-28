import gradio as gr
from gradio_rangeslider import RangeSlider
from src.backend.parameterSelectionUtils import *

class ParameterSelection:
    def __init__(self, output_component, examples_data):
        self.output_component = output_component
        self.init_components()

    def init_components(self):
        # Original Gradio components from ParameterSelection section
        self.species_dropdown = gr.Dropdown(choices=species_update_list(), value="Custom", label="Species", allow_custom_value=True, render=False, interactive=True)        
        self.species_json_viewer = gr.JSON(label="Parameters", render=False)  # JSON viewer for non-custom selections
        self.species_json_editor = gr.Textbox(label="Editable parameters", lines=20, visible=False, render=False, interactive=True)  # Hidden by default
        self.species_editor_save_text = gr.Textbox(label="Species name", visible=False, render=False, interactive=True)
        self.species_editor_save_button = gr.Button("Save", visible=False, render=False, interactive=True)
        self.morphologies_checkboxes = gr.CheckboxGroup(choices=get_species_dict().keys(), label="Morphologies", render=False, interactive=True) 
        self.debris_checkbox = gr.Checkbox(label="Enable debris generation", render=False, interactive=True)
        
    def read_examples(self, examples_data, index):
        example = examples_data["parameter_selection"][index]
        return [
            example["species"],
            [int(element) for element in example["n_spermatozoa"]],
            bool(example["debris_rendering"]),
            [int(element) for element in example["n_debris"]]
        ]
        
    def set_values(self, examples_data, index):
        examples = self.read_examples(examples_data, index)
        return [
            gr.update(value=examples[0]),
            gr.update(value=examples[1]),
            gr.update(value=examples[2]),
            gr.update(value=examples[3])
        ]
        
    def get_values(self):
        return [
            self.species_dropdown.value,
            self.morphologies_checkboxes.value,
            self.debris_checkbox.value,
            self.n_debris.value
        ]  
        
    def render(self):
        with gr.Group():
            with gr.Accordion("Species configuration", open=True):
                self.species_dropdown.render()
                with gr.Accordion("Settings explorer", open=False):
                    self.species_json_viewer.render()
                    self.species_json_editor.render()
                with gr.Column():
                    self.species_editor_save_text.render()
                    self.species_editor_save_button.render()
            with gr.Accordion("Cell morphology and count", open=True):
                self.morphologies_checkboxes.render()
                self.n_spermatozoa = RangeSlider(label="N spermatozoa", minimum=0, maximum=250)
                self.debris_checkbox.render()
                self.n_debris = RangeSlider(label="N debris", minimum=0, maximum=500)
