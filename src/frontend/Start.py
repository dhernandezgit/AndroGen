
import gradio as gr

class Start:
    def __init__(self):
        self.init_components()

    def init_components(self):
        # Original Gradio components from Welcome section
        self.welcome_message = gr.Markdown(
            """
            # AndroGen App
            
            **AndroGen** is a application designed to generate synthetic images of sperm cells, with a user-friendly interface and customizable parameters.
            
            ### How to Use the App

            - Explore different *Image appearances* and *Cell settings* using the **Presets**.
            - Click **Generate sample** to preview how your current settings produce synthetic sperm images.
            - Adjust the dataset parameters and use **Create dataset** to generate your own custom synthetic dataset.
            - Optionally, you can **Save custom preset** to reuse your current configuration later. 

            ### Key Benefits

            - *Ready-to-use annotations*: Our synthetic image includes labels in both **YOLO format** and **segmentation masks**, making your dataset training-ready from the start.
            - *Zero-shot generation*: No need for pre-training or data preparation. Start generating synthetic sperm cell images immediately with configurable parameters and preloaded species with different morphological classes.
            - *Fast and flexible dataset creation*: Define your custom dataset settings in minutes and start the process to get thousands of realistic annotated images in just a few hours.
                        
            ### Learn More
            For a detailed explanation of how this app works, please refer to the original research paper: ***AndroGen: Open-Source Synthetic Data Generation for Automated Sperm Analysis***. We hope AndroGen helps accelerate your research and development in the field!
            """,
            render=False,
        )

    def render(self):
        with gr.Blocks():
            self.welcome_message.render()
