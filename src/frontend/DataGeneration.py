import os
import gradio as gr
from gradio_imageslider import ImageSlider

class DataGeneration:
    def __init__(self, examples_data):
        self.init_components()

    def read_examples(self, examples_data, index):
        example = examples_data["data_generation"][index]
        return []

    def init_components(self):
        # Original Gradio components from DataGeneration section
        self.output = gr.Gallery(label="Output", preview=False, columns=3, interactive=False, render=False) #selected_index=0,
        #self.output = ImageSlider(value=["data/quality_score/real-test/9.png", "data/quality_score/new_synthetic/frames/000000_000003.png"], label="Output", interactive=False, render=False)
        self.generate_button = gr.Button("Generate sample", render=False)
        self.create_dataset_button = gr.Button("Create dataset", render=False)
        self.dataset_name = gr.Textbox(value="SyntheticDataset", label="Dataset Name", placeholder="Enter the dataset name here", interactive=True, render=False)
        self.save_folder = gr.Textbox(value=os.getcwd(),label="Save Folder", placeholder="Enter the folder path to save images", interactive=True, render=False)
        self.text_n_sequences = gr.Textbox(value="1000", label="Number of sequences", interactive=True, render=False)
        self.text_n_frames_sequence = gr.Textbox(value="25", label="Number of frames per sequence", interactive=True, render=False)
        self.text_seed = gr.Textbox(value="42", label="Generator seed", interactive=True, render=False)
        self.remaining_time_box = gr.Markdown(render=False)
        
    def render(self):
        with gr.Group():
            self.output.render()
            self.generate_button.render()
                
            with gr.Accordion("Create synthetic dataset", open=False):
                with gr.Row():
                    self.dataset_name.render()
                    self.save_folder.render()
                with gr.Row():
                    self.text_n_sequences.render()
                    self.text_n_frames_sequence.render()
                    self.text_seed.render()
                self.create_dataset_button.render()