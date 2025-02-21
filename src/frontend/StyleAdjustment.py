import os
import sys
sys.path.append(os.path.abspath('.'))

import gradio as gr

from src.backend.utils import read_json, get_color

class StyleAdjustment:
    def __init__(self, output_component, examples_data):
        self.output_component = output_component
        self.init_components()

    def read_examples(self, examples_data, index):
        example = examples_data["style_adjustment"][index]
        return [
            [os.path.join(example["files"], p) for p in os.listdir(example["files"])],
            example["filter_images"],
            float(example["contrast"]),
            float(example["brightness"]),
            example["horizontal_flip"],
            example["vertical_flip"],
            example["annotation"],
            example["spermatozoon_head_color"],
            example["spermatozoon_head_highlight_color"],
            example["spermatozoon_neck_color"],
            example["spermatozoon_tail_color"],
            example["spermatozoon_droplet_color"],
            example["debris_color"],
            example["shadow_start_color"],
            example["shadow_end_color"],
            int(example["spermatozoon_scale_slider"]),
            int(example["shadow_offset_slider"]),
            int(example["shadow_scale_slider"]),
            int(example["n_points_render_slider"])
        ]
        
    def init_components(self):
        # Original Gradio components from StyleAdjustment section
        self.input_images = gr.File(label="Upload Images", file_types=["image"], type="filepath", file_count="multiple", elem_id="file-uploader", interactive=True, render=False)
        self.filter_images = gr.Checkbox(label="Filter background images", render=False, interactive=True)
        self.background_button = gr.Button("Generate test backgrounds", render=False, interactive=True)
        self.background_output = gr.Gallery(label="Sample backgrounds", preview=False, columns=3, render=False, interactive=False)
        self.contrast_variation = gr.Slider(label="Contrast deviation", minimum=0.0, maximum=1.0, render=False, interactive=True)
        self.brightness_variation = gr.Slider(label="Brightness deviation", minimum=0.0, maximum=1.0, render=False, interactive=True)
        self.horizontal_flip_checkbox = gr.Checkbox(label="Horizontal flip", render=False, interactive=True)
        self.vertical_flip_checkbox = gr.Checkbox(label="Vertical flip", render=False, interactive=True)
        
        self.color_image_input = gr.Image(type="numpy", label="Upload your reference image", interactive=True, render=False)
        self.color_display = gr.ColorPicker(label="Colour Display", render=False, interactive=True)
        self.get_color_button = gr.Button("Get colour", render=False, interactive=True)
        
        self.spermatozoon_head_color = gr.ColorPicker(label="Head", render=False, interactive=True)
        self.spermatozoon_head_highlight_color = gr.ColorPicker(label="Head highlight", render=False, interactive=True)
        self.spermatozoon_neck_color = gr.ColorPicker(label="Neck", render=False, interactive=True)
        self.spermatozoon_tail_color = gr.ColorPicker(label="Tail", render=False, interactive=True)
        self.spermatozoon_droplet_color = gr.ColorPicker(label="Droplet", render=False, interactive=True)
        self.debris_color = gr.ColorPicker(label="Debris", render=False, interactive=True)
        self.shadow_start_color = gr.ColorPicker(label="Shadow start", render=False, interactive=True)
        self.shadow_end_color = gr.ColorPicker(label="Shadow end", render=False, interactive=True)
        self.spermatozoon_scale_slider = gr.Slider(label="Sperm scale", minimum=0.1, maximum=100, render=False, interactive=True)
        self.shadow_offset_slider = gr.Slider(label="Shadow offset", minimum=-10, maximum=10, render=False, interactive=True)
        self.shadow_scale_slider = gr.Slider(label="Shadow scale", minimum=0, maximum=10, render=False, interactive=True)
        self.n_points_render_slider = gr.Slider(label="N points to render", minimum=10, maximum=1000, render=False, interactive=True)
        
    def render(self):
        with gr.Group():
            with gr.Accordion("Background generation", open=True):
                self.input_images.render()
                self.background_button.render()
                self.background_output.render()
                self.filter_images.render()
                
            with gr.Accordion("Image augmentation settings", open=False):
                self.contrast_variation.render()
                self.brightness_variation.render()
                self.horizontal_flip_checkbox.render()
                self.vertical_flip_checkbox.render()
                
            with gr.Accordion("Colour picking options", open=False):
                self.color_image_input.render()
                
                gr.Markdown(" ### Spermatozoon colours")
                with gr.Row():
                    self.spermatozoon_head_color.render()
                    self.spermatozoon_head_highlight_color.render()
                    self.spermatozoon_neck_color.render()
                    self.spermatozoon_tail_color.render()
                    self.spermatozoon_droplet_color.render()
                    
                gr.Markdown(" ### Shadow an debris colours")
                with gr.Row():
                    self.debris_color.render()
                    self.shadow_start_color.render()
                    self.shadow_end_color.render()
                    
                    
            with gr.Accordion("Advanced style parameters", open=False):
                gr.Markdown(" ### Scales and other parameters")
                with gr.Row():
                    self.spermatozoon_scale_slider.render()
                    self.shadow_offset_slider.render()
                    self.shadow_scale_slider.render()
                    self.n_points_render_slider.render()
                    
            self.get_color_button.click(get_color, inputs=[self.color_image_input], outputs=[self.color_display])