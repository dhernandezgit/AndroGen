
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
            - Use the **examples** to explore different dataset settings and styles.
            - Visit the **Data Generation** section to experiment with generating synthetic images.
            - Make use of **advanced settings** to fine-tune the output to meet your specific requirements.

            ### Key Benefits
            - Eliminate the need for labor-intensive labeling of real data.
            - Leverage high-quality synthetic data for robust model training.
            
            ### Learn More
            For a detailed explanation of how this app works, please refer to the original research paper.

            We hope AndroGen helps accelerate your research and development in the field!
            """,
            render=False,
        )

    def render(self):
        with gr.Blocks():
            self.welcome_message.render()
