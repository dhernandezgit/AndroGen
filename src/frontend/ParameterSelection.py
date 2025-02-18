import gradio as gr
from gradio_rangeslider import RangeSlider

class ParameterSelection:
    def __init__(self, output_component, examples_data):
        self.output_component = output_component
        self.init_components()

    def read_examples(self, examples_data, index):
        example = examples_data["parameter_selection"][index]
        return [
            example["species"],
            [int(element) for element in example["n_spermatozoa"]],
            bool(example["debris_rendering"]),
            [int(element) for element in example["n_debris"]],
            bool(example["same_probabilities"])
        ]

    def init_components(self):
        # Original Gradio components from ParameterSelection section
        self.species_dropdown = gr.Dropdown(label="Species", allow_custom_value=True, render=False, interactive=True)        
        self.species_json_viewer = gr.JSON(label="Parameters", render=False)  # JSON viewer for non-custom selections
        self.species_json_editor = gr.Textbox(label="Editable parameters", lines=20, visible=False, render=False, interactive=True)  # Hidden by default
        self.species_editor_save_text = gr.Textbox(label="Species name", visible=False, render=False, interactive=True)
        self.species_editor_save_button = gr.Button("Save", visible=False, render=False, interactive=True)
        self.morphologies_checkboxes = gr.CheckboxGroup(label="Morphologies", render=False, interactive=True) 
        self.debris_checkbox = gr.Checkbox(label="Debris rendering", render=False, interactive=True)
        self.same_probabilities_checkbox = gr.Checkbox(label="Use same probabilities for each class", render=False, interactive=True)

    def render(self):
        with gr.Group():
            with gr.Accordion("Measurement JSON", open=True):
                self.species_dropdown.render()
                with gr.Accordion("JSON viewer", open=False):
                    self.species_json_viewer.render()
                    self.species_json_editor.render()
                with gr.Column():
                    self.species_editor_save_text.render()
                    self.species_editor_save_button.render()
            with gr.Accordion("Cell morphology and count", open=True):
                self.morphologies_checkboxes.render()
                gr.Markdown("### Number of elements")
                self.same_probabilities_checkbox.render()
                self.n_spermatozoa = RangeSlider(label="N spermatozoa", minimum=0, maximum=250)
                self.debris_checkbox.render()
                self.n_debris = RangeSlider(label="N debris", minimum=0, maximum=500)
