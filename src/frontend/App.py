import os
import sys
sys.path.append(os.path.abspath('.'))
import configparser
import shutil
import numpy as np
import cv2

import gradio as gr
from src.frontend.Start import Start
from src.frontend.StyleAdjustment import StyleAdjustment
from src.frontend.ParameterSelection import ParameterSelection
from src.frontend.DataGeneration import DataGeneration

from src.backend.parameterSelectionUtils import *
from src.backend.styleAdjustmentUtils import get_color, process_color
from src.backend.dataGenerationUtils import *

from src.backend.SpermatozoonFactory import SpermatozoonFactory
from src.backend.DebrisFactory import DebrisFactory
from src.backend.BackgroundGenerator import BackgroundGenerator
from src.backend.ConfigParser import SequenceConfig, DatasetConfig
from src.backend.SequenceGenerator import SequenceGenerator
from src.backend.DatasetMaker import DatasetMaker
from src.backend.ImageAugmentor import ImageAugmentor
from src.backend.utils import save_gif, read_json, update_json, write_json

class App:
    def __init__(self, config_path):
        self.config_path = config_path
        
        config = configparser.ConfigParser()
        config.read(self.config_path)
        self.test_sequence_path = config['Paths']['test_sequence_path']
        self.test_gif_path = config['Paths']['test_gif_path']
        self.species_dict_path = config['Paths']['species_dict_path']
        self.style_config_path = config['Paths']['style_config_path']
        self.debris_dict_path = config['Paths']['debris_dict_path']
        self.sequence_config = SequenceConfig(config_path)
        self.dataset_config = DatasetConfig(config_path)
        set_species_path("/".join(self.species_dict_path.split("/")[:-1]))
        set_species_selection_path(self.species_dict_path.split("/")[-1].split(".")[0])
        self.species_selection_dict_path = get_species_selection_path()
        os.makedirs(os.path.join(self.test_sequence_path, "frames"), exist_ok=True)
        os.makedirs(os.path.join(self.test_sequence_path, "labels"), exist_ok=True)
        
        self.bgg = BackgroundGenerator()
        self.sf = SpermatozoonFactory(self.species_selection_dict_path, self.style_config_path)
        self.df = DebrisFactory(self.debris_dict_path, self.style_config_path)
        
        self.examples_data = read_json(os.path.join('cfg', 'examples.json'))
        self.start = Start()
        self.data_generation = DataGeneration(self.examples_data)
        self.style_adjustment = StyleAdjustment(self.data_generation.output, self.examples_data)
        self.parameter_selection = ParameterSelection(self.data_generation.output, self.examples_data)
        self.examples_processed = [self.style_adjustment.read_examples(self.examples_data, i)+
                                   self.parameter_selection.read_examples(self.examples_data, i)+
                                   self.data_generation.read_examples(self.examples_data, i) for i, name in enumerate(self.examples_data["names"])]
        self.example_text = gr.Textbox(placeholder="Preset 1", label="Custom preset name", interactive=True, render=False, visible=False)
        self.examples_save_button = gr.Button("Save current configuration as preset 💾", render=False, interactive=True, visible=False)
        self.examples_markdown = gr.Markdown("# Click on an example to start", render=False, visible=True)
        self.timer = gr.Timer(active=False, value=1, render=False)
        self.tick_state = gr.State(value=False, render=False)
        self.init_components()
        
    def _set_values(self, i=0):
        return self.style_adjustment.set_values(self.examples_data, i) + self.parameter_selection.set_values(self.examples_data, i) + self.data_generation.set_values(self.examples_data, i)
            
    def _init_inputs(self, index=0):
        self.input_list = [self.style_adjustment.input_images,
                        self.style_adjustment.filter_images,
                        self.style_adjustment.contrast_variation,
                        self.style_adjustment.brightness_variation,
                        self.style_adjustment.blur_variation,
                        self.style_adjustment.horizontal_flip_checkbox,
                        self.style_adjustment.vertical_flip_checkbox,
                        self.style_adjustment.color_image_input,
                        self.style_adjustment.color_image_input,
                        self.style_adjustment.spermatozoon_head_color,
                        self.style_adjustment.spermatozoon_head_highlight_color,
                        self.style_adjustment.spermatozoon_neck_color,
                        self.style_adjustment.spermatozoon_tail_color,
                        self.style_adjustment.spermatozoon_droplet_color,
                        self.style_adjustment.debris_color,
                        self.style_adjustment.shadow_start_color,
                        self.style_adjustment.shadow_end_color,
                        self.style_adjustment.spermatozoon_scale_slider,
                        self.style_adjustment.debris_scale_slider,
                        self.style_adjustment.shadow_offset_slider,
                        self.style_adjustment.shadow_scale_slider,
                        self.style_adjustment.n_points_render_slider,
                        self.style_adjustment.z_range,
                        self.parameter_selection.species_dropdown,
                        self.parameter_selection.n_spermatozoa,
                        self.parameter_selection.debris_checkbox,
                        self.parameter_selection.n_debris,
                        self.data_generation.text_n_frames_sequence,
                        self.data_generation.text_n_sequences
                        ]

        #for input_element, value in zip(self.input_list, self.examples_processed[index]):
        #    input_element.value = value

    def _save_example(self, example_name, *input_list):
        self.examples_data = read_json(os.path.join('cfg', 'examples.json')) 
        #self.style_adjustment.set_values(self.examples_data, i) + self.parameter_selection.set_values(self.examples_data, i) + self.data_generation.set_values(self.examples_data, i)
        indexes = [0, len(self.examples_data["style_adjustment"][0])-1, len(self.examples_data["parameter_selection"][0]), len(self.examples_data["data_generation"][0])]
        style_adjustment_example = input_list[indexes[0]:indexes[1]]
        parameter_selection_example = input_list[indexes[1]:indexes[1]+indexes[2]]
        data_generation_example = input_list[indexes[1]+indexes[2]:indexes[1]+indexes[2]+indexes[3]]
        
        generated_path = os.path.join("resources/examples/predefined_generated_images",f"S{example_name}.png")
        real_path = os.path.join("resources/examples/predefined_background_file",f"{example_name}.png")
        bg_path = os.path.join("resources/examples/predefined_background", example_name)
        
        shutil.copy("resources/sample/frames/000000_000000.png", generated_path)
        cv2.imwrite(real_path, cv2.cvtColor(style_adjustment_example[7], cv2.COLOR_BGR2RGB))
        os.makedirs(bg_path, exist_ok=True)
        for f in style_adjustment_example[0]:
            shutil.copy(f, os.path.join(bg_path, f.split("/")[-1]))
        style_adjustment_example = (bg_path,) + style_adjustment_example[1:7] + (real_path, generated_path,) + style_adjustment_example[9:]
        
        self.examples_data["names"].append(example_name)
        style_adjustment_example_dict = {}
        for key, value in zip(self.examples_data["style_adjustment"][0].keys(), style_adjustment_example):
            style_adjustment_example_dict[key] = value
        style_adjustment_example_dict["active_classes"] = "Normal"
        self.examples_data["style_adjustment"].append(style_adjustment_example_dict)
        parameter_selection_example_dict = {}
        for key, value in zip(self.examples_data["parameter_selection"][0].keys(), parameter_selection_example):
            parameter_selection_example_dict[key] = value
        self.examples_data["parameter_selection"].append(parameter_selection_example_dict)
        data_generation_example_dict = {}
        for key, value in zip(self.examples_data["data_generation"][0].keys(), data_generation_example):
            data_generation_example_dict[key] = int(value)
        self.examples_data["data_generation"].append(data_generation_example_dict)
        write_json(os.path.join('cfg', 'examples.json'), self.examples_data)
        
        #images = [(example[7], self.examples_data["names"][i]) for i, example in enumerate(self.examples_processed)]
        return [(example["annotation"], self.examples_data["names"][i]) for i, example in enumerate(self.examples_data["style_adjustment"])]
    
    def _example_select(self, evt: gr.SelectData):
        index = evt.index  # Get index of clicked image
        values = self._set_values(i=index)
        return values + [gr.update(scale = 4), gr.update(visible=True), gr.update(visible = True), gr.update(visible = True), gr.update(visible = True), gr.update(columns = [7]), gr.update(visible = True), gr.update(visible = True), gr.update(visible = False), gr.update(active = True)]
        
    def _update_background_generator(self, images, contrast, brightness, horizontal_flip, vertical_flip, n_images_out=9):
        self.sequence_config.update()
        if len(images) < 1:  # Check if the list of images is empty
            raise gr.Error("Please select at least one image.")
        elif len(images) == 1:  # If there is only one image, use it as the background
            self.bgg.setGenerationMethod('single', paths=images[0])
            bg = self.bgg.getBackground(resolution=self.sequence_config.getParameters()['Sequence.resolution']['resolution'])
            ia = ImageAugmentor(contrast=contrast, brightness=brightness, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip)
            return ia.augment(bg, num_images=n_images_out)
        else:  # If there are multiple images, use them as a background
            if self.style_adjustment.filter_images:
                self.bgg.setGenerationMethod('list', paths=images)
                ia = ImageAugmentor(contrast=contrast, brightness=brightness, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip)
                images_out = []
                for n in range(n_images_out):
                    bg = self.bgg.getBackground(resolution=self.sequence_config.getParameters()['Sequence.resolution']['resolution'])
                    images_out.append(ia.augment(bg, num_images=1)[0][0])
                return images_out
            else:
                images_out = []
                for n in range(n_images_out):
                    image = np.random.choice(images, 1)[0]
                    self.bgg.setGenerationMethod('single', single_image_path=image)
                    ia = ImageAugmentor(contrast=contrast, brightness=brightness, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip)
                    images_out.append(ia.augment(self.bgg.getBackground(resolution=self.sequence_config.getParameters()['Sequence.resolution']['resolution']), num_images=1)[0][0])
                return images_out
        
    def _update_config(self, name, value):
        if isinstance(value, tuple):  # Check if value is a tuple
            value = list(value)
        
        params = {}
        params["Sequence.quantities"] = {}
        params["Sequence.quantities"][name] = str(value)
        self.sequence_config.writeParameters(params)
        
    def _update_augmentation(self, contrast, brightness, horizontal_flip, vertical_flip):
        params = {}
        params['Sequence.augmentation'] = {}
        params['Sequence.augmentation']['contrast'] = contrast
        params['Sequence.augmentation']['brightness'] = brightness
        params['Sequence.augmentation']['horizontal_flip'] = horizontal_flip
        params['Sequence.augmentation']['vertical_flip'] = vertical_flip 
        self.sequence_config.writeParameters(params)
                
    def _update_colors(self, head_color, head_highlight_color, neck_color, tail_color, droplet_color, debris_color, shadow_start_color, shadow_end_color, scale, debris_scale, shadow_offset, shadow_scale, n_points, z_range, blur, active_classes):
        params = read_json(self.style_config_path)
        params['color']['head']['r'] = process_color(head_color)[0]
        params['color']['head']['g'] = process_color(head_color)[1]
        params['color']['head']['b'] = process_color(head_color)[2]
        params['color']['head_highlight']['r'] = process_color(head_highlight_color)[0]
        params['color']['head_highlight']['g'] = process_color(head_highlight_color)[1]
        params['color']['head_highlight']['b'] = process_color(head_highlight_color)[2]
        params['color']['neck']['r'] = process_color(neck_color)[0]
        params['color']['neck']['g'] = process_color(neck_color)[1]
        params['color']['neck']['b'] = process_color(neck_color)[2]
        params['color']['tail']['r'] = process_color(tail_color)[0]
        params['color']['tail']['g'] = process_color(tail_color)[1]
        params['color']['tail']['b'] = process_color(tail_color)[2]
        params['color']['droplet']['r'] = process_color(droplet_color)[0]
        params['color']['droplet']['g'] = process_color(droplet_color)[1]
        params['color']['droplet']['b'] = process_color(droplet_color)[2]
        params['color']['debris']['r'] = process_color(debris_color)[0]
        params['color']['debris']['g'] = process_color(debris_color)[1]
        params['color']['debris']['b'] = process_color(debris_color)[2]
        params['color']['shadow_start']['r'] = process_color(shadow_start_color)[0]
        params['color']['shadow_start']['g'] = process_color(shadow_start_color)[1]
        params['color']['shadow_start']['b'] = process_color(shadow_start_color)[2]
        params['color']['shadow_end']['r'] = process_color(shadow_end_color)[0]
        params['color']['shadow_end']['g'] = process_color(shadow_end_color)[1]
        params['color']['shadow_end']['b'] = process_color(shadow_end_color)[2]
        params['color']['debris_shadow_start']['r'] = process_color(shadow_start_color)[0]
        params['color']['debris_shadow_start']['g'] = process_color(shadow_start_color)[1]
        params['color']['debris_shadow_start']['b'] = process_color(shadow_start_color)[2]
        params['color']['debris_shadow_end']['r'] = process_color(shadow_end_color)[0]
        params['color']['debris_shadow_end']['g'] = process_color(shadow_end_color)[1]        
        params['color']['debris_shadow_end']['b'] = process_color(shadow_end_color)[2]
        params['scale'] = scale
        params['debris_scale'] = debris_scale
        params['shadow_offset'] = shadow_offset
        params['shadow_starting_scale'] = max(int(shadow_scale/4), 1)
        params['shadow_ending_scale'] = shadow_scale
        params['debris_shadow_offset'] = shadow_offset
        params['debris_shadow_starting_scale'] = max(int(shadow_scale/2), 1)
        params['debris_shadow_ending_scale'] = shadow_scale
        params['n_points'] = n_points
        params['z_start'] = z_range[0]
        params['z_end'] = z_range[1]
        params['blur_start'] = blur[0]
        params['blur_end'] = blur[1]
        params['active_classes'] = active_classes
        update_json(self.style_config_path, params)
        self.species_selection_dict_path = get_species_selection_path()
        self.sf = SpermatozoonFactory(self.species_selection_dict_path, self.style_config_path)
        self.df = DebrisFactory(self.debris_dict_path, self.style_config_path)
           
    def _species_select(self, selection, custom_json):
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
            set_species_selection_path(selection)
            set_species_dict(read_json(get_species_selection_path()))
            self.species_selection_dict_path = get_species_selection_path()
            self.sf = SpermatozoonFactory(self.species_selection_dict_path, self.style_config_path)
            self.df = DebrisFactory(self.debris_dict_path, self.style_config_path)
            return [
        gr.update(choices=species_update_list()),
        gr.update(choices=list(get_species_dict().keys()), value=list(get_species_dict().keys())[0]),
        gr.update(visible=False),
        gr.update(visible=True, value=get_species_dict()),
        gr.update(visible=False), 
        gr.update(visible=False)
        ]
           
    def _generate_sequence(self, n_frames, extra_images = 9):
        n_frames = int(n_frames)
        remove_old=True
        if remove_old:
            if os.path.isdir(os.path.join(self.test_sequence_path, "frames")):
                shutil.rmtree(os.path.join(self.test_sequence_path, "frames"))
            if os.path.exists(self.test_gif_path):
                os.remove(self.test_gif_path)
        params = self.sequence_config.getParameters()['Sequence.augmentation']
        ia = ImageAugmentor(contrast=params['contrast'], brightness=params['brightness'], horizontal_flip=params['horizontal_flip'], vertical_flip=params['vertical_flip'])
        sg = SequenceGenerator(n_frames, self.sequence_config, self.sf, self.df, self.bgg)
        progress = gr.Progress(track_tqdm=True)
        for _ in progress.tqdm(sg.generate_sequence(output_dir=self.test_sequence_path, yield_progress=True), total=n_frames, desc="Generating synthetic frames..."):
            ...
            
        save_gif(os.path.join(self.test_sequence_path, "frames"), 0)
        if n_frames > 1:
            return self.test_gif_path# + [os.path.join(self.test_sequence_path, "frames", p) for p in np.random.choice(os.listdir(os.path.join(self.test_sequence_path, "frames")), extra_images)]
        else:
            return [os.path.join(self.test_sequence_path, "frames", p) for p in os.listdir(os.path.join(self.test_sequence_path, "frames"))][0]
        
        
    def _tick(self, state):
        return [not state, gr.update(active = False)]
    
    def _generate_sequence_timer(self, n_frames, extra_images = 9):
        return [self._generate_sequence(n_frames=n_frames, extra_images = 9, yield_progress=True)]
        
    def _generate_dataset(self, dataset_name, save_folder, n_sequences, n_frames_sequence, seed=42):
        #np.random.seed(int(seed))
        #random.seed(int(seed))
        params = self.sequence_config.getParameters()['Sequence.augmentation']
        ia = ImageAugmentor(contrast=params['contrast'], brightness=params['brightness'], horizontal_flip=params['horizontal_flip'], vertical_flip=params['vertical_flip'])
        dm = DatasetMaker(num_sequences=int(n_sequences), num_frames=int(n_frames_sequence), sequence_config=self.sequence_config, sf=self.sf, df=self.df, bgg=self.bgg, output_dir=os.path.join(save_folder, dataset_name), image_augmentor=ia)
        progress = gr.Progress(track_tqdm=True)
        for _ in progress.tqdm(dm.generate_dataset(), total=n_sequences, desc="Generating synthetic dataset..."):
            ...
    
    def init_components(self):
        custom_css = """
        h1 {
            text-align: center;
            display:block;
        }
        #file-uploader {
            max-height: 150px; /* Adjust the height as needed */
            overflow-y: auto; /* Enable scrolling if content overflows */
            border: 1px solid #ccc; /* Optional: Add a border for better visibility */
            padding: 5px; /* Optional: Add some padding for better spacing */
        }
        #examples-container {
            background-color: #d67f29; /* Coral orange background for Examples */
            padding: 5px;
            
        """
        
        js_func = """
        function refresh() {
            const url = new URL(window.location);

            if (url.searchParams.get('__theme') !== 'dark') {
                url.searchParams.set('__theme', 'dark');
                window.location.href = url.href;
            }
        }
        """
        with gr.Blocks(css=custom_css, fill_width=True, delete_cache=(60, 1200), fill_height=True, theme=gr.themes.Soft()) as self.app:
            with gr.Row():
                #with gr.Sidebar(width="35%", open=False) as self.col_left:
                with gr.Column(scale=999999999) as self.col_left:
                    with gr.Tabs(visible=False) as self.col_left_group:
                        with gr.TabItem("Start") as tab_start:
                            self.start.render()
                        with gr.TabItem("Image Appearance") as tab_style_adjustment:
                            self.style_adjustment.render()
                        with gr.TabItem("Cell Settings") as tab_parameter_selection:
                            self.parameter_selection.render()
                    self._init_inputs(index=0)
                    with gr.Accordion("", open=True) as self.examples_accordion:
                        images = [(example[7], self.examples_data["names"][i]) for i, example in enumerate(self.examples_processed)]
                        self.examples_markdown.render()
                        self.examples_gallery = gr.Gallery(value=images, type="pil", label="Presets", elem_id="examples-container", columns=3, interactive=True, height="10%", allow_preview=False)
                        self.timer.render()
                        self.tick_state.render()
                        with gr.Column(scale=1):
                            self.example_text.render()
                            self.examples_save_button.render()
                    #with gr.Group():
                    #    with gr.Row():
                    #        with gr.Column(scale=1):
                                #self.examples = gr.Examples(self.examples_processed, elem_id="examples-container", label="Dataset examples", example_labels=self.examples_data["names"],
                                #inputs=self.input_list,
                                #outputs=None,
                                #postprocess=True,
                                #examples_per_page=15)
                                #self.examples.load_input_event.then(self._generate_sequence, [self.data_generation.text_n_frames_sequence], [self.data_generation.output])
                                #processed_example = self.examples._get_processed_example(self.examples_processed[0])
                                #print(processed_example)
                                #self.app.load(lambda: ([gr.update(input_elem, value=example_value) for input_elem, example_value in zip(self.input_list, self.examples_processed[0])]), None, self.input_list)
                    #        with gr.Column(scale=1):
                    #            self.example_text.render()
                    #            self.examples_save_button.render()
                with gr.Column(scale=4) as self.col_right:
                    self.data_generation.render()

            self.examples_gallery.select(self._example_select, None, [*self.input_list, self.col_left, self.col_left_group, self.data_generation.output, self.data_generation.generate_button, self.data_generation.advanced_settings, self.examples_gallery, self.example_text, self.examples_save_button, self.examples_markdown, self.timer])
            self.examples_save_button.click(self._save_example, [self.example_text, *self.input_list], [self.examples_gallery])
            self.timer.tick(self._tick, [self.tick_state], [self.tick_state, self.timer])
            self.tick_state.change(self._generate_sequence, [self.data_generation.text_n_frames_sequence], [self.data_generation.output])
            tab_start.select(lambda: (gr.update(scale=4)), None, [self.col_left])
            tab_style_adjustment.select(lambda: (gr.update(scale=4)), None, [self.col_left])
            tab_parameter_selection.select(lambda: (gr.update(scale=4)), None, [self.col_left])
            
            self.style_adjustment.background_button.click(self._update_background_generator, [self.style_adjustment.input_images, self.style_adjustment.contrast_variation, self.style_adjustment.brightness_variation, self.style_adjustment.horizontal_flip_checkbox, self.style_adjustment.vertical_flip_checkbox], [self.style_adjustment.background_output])
            self.style_adjustment.input_images.change(self._update_background_generator, [self.style_adjustment.input_images, self.style_adjustment.contrast_variation, self.style_adjustment.brightness_variation, self.style_adjustment.horizontal_flip_checkbox, self.style_adjustment.vertical_flip_checkbox], [self.style_adjustment.background_output])
            self.style_adjustment.filter_images.change(self._update_background_generator, [self.style_adjustment.input_images, self.style_adjustment.contrast_variation, self.style_adjustment.brightness_variation, self.style_adjustment.horizontal_flip_checkbox, self.style_adjustment.vertical_flip_checkbox], [self.style_adjustment.background_output])
            
            self.style_adjustment.contrast_variation.change(self._update_augmentation, [self.style_adjustment.contrast_variation, self.style_adjustment.brightness_variation, self.style_adjustment.horizontal_flip_checkbox, self.style_adjustment.vertical_flip_checkbox], [])
            self.style_adjustment.brightness_variation.change(self._update_augmentation, [self.style_adjustment.contrast_variation, self.style_adjustment.brightness_variation, self.style_adjustment.horizontal_flip_checkbox, self.style_adjustment.vertical_flip_checkbox], [])
            self.style_adjustment.horizontal_flip_checkbox.change(self._update_augmentation, [self.style_adjustment.contrast_variation, self.style_adjustment.brightness_variation, self.style_adjustment.horizontal_flip_checkbox, self.style_adjustment.vertical_flip_checkbox], [])
            self.style_adjustment.vertical_flip_checkbox.change(self._update_augmentation, [self.style_adjustment.contrast_variation, self.style_adjustment.brightness_variation, self.style_adjustment.horizontal_flip_checkbox, self.style_adjustment.vertical_flip_checkbox], [])
            
            self.style_adjustment.spermatozoon_head_color.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            self.style_adjustment.spermatozoon_head_highlight_color.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            self.style_adjustment.spermatozoon_neck_color.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            self.style_adjustment.spermatozoon_head_highlight_color.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            self.style_adjustment.spermatozoon_neck_color.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            self.style_adjustment.spermatozoon_tail_color.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            self.style_adjustment.spermatozoon_droplet_color.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            self.style_adjustment.debris_color.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            self.style_adjustment.shadow_start_color.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            self.style_adjustment.shadow_end_color.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            self.style_adjustment.spermatozoon_scale_slider.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            self.style_adjustment.shadow_offset_slider.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            self.style_adjustment.shadow_scale_slider.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            self.style_adjustment.n_points_render_slider.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            self.style_adjustment.blur_variation.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            self.style_adjustment.z_range.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            
            self.parameter_selection.species_dropdown.change(self._species_select, [self.parameter_selection.species_dropdown, self.parameter_selection.species_json_viewer], [self.parameter_selection.species_dropdown, self.parameter_selection.morphologies_checkboxes, self.parameter_selection.species_json_editor, self.parameter_selection.species_json_viewer, self.parameter_selection.species_editor_save_text, self.parameter_selection.species_editor_save_button])
            self.parameter_selection.species_editor_save_button.click(save_custom_json, [self.parameter_selection.species_json_editor, self.parameter_selection.species_editor_save_text], [self.parameter_selection.species_dropdown])
    
            self.parameter_selection.morphologies_checkboxes.change(self._update_colors, [self.style_adjustment.spermatozoon_head_color, self.style_adjustment.spermatozoon_head_highlight_color, self.style_adjustment.spermatozoon_neck_color, self.style_adjustment.spermatozoon_tail_color, self.style_adjustment.spermatozoon_droplet_color, self.style_adjustment.debris_color, self.style_adjustment.shadow_start_color, self.style_adjustment.shadow_end_color, self.style_adjustment.spermatozoon_scale_slider, self.style_adjustment.debris_scale_slider, self.style_adjustment.shadow_offset_slider, self.style_adjustment.shadow_scale_slider, self.style_adjustment.n_points_render_slider, self.style_adjustment.z_range, self.style_adjustment.blur_variation, self.parameter_selection.morphologies_checkboxes], [])
            self.parameter_selection.n_spermatozoa.change(self._update_config, [gr.State("spermatozoon_n"), self.parameter_selection.n_spermatozoa], None)
            self.parameter_selection.debris_checkbox.change(self._update_config, [gr.State("render_debris"), self.parameter_selection.debris_checkbox], None)
            self.parameter_selection.n_debris.change(self._update_config, [gr.State("debris_n"), self.parameter_selection.n_debris], None)
                
            self.data_generation.generate_button.click(self._generate_sequence, [self.data_generation.text_n_frames_sequence], [self.data_generation.output])
            self.data_generation.create_dataset_button.click(self._generate_dataset, [self.data_generation.dataset_name, self.data_generation.save_folder, self.data_generation.text_n_sequences, self.data_generation.text_n_frames_sequence], [self.data_generation.output]) #, self.data_generation.text_seed
    
    def launch(self):    
        self.app.launch(allowed_paths=["/media/daniel/TOSHIBA_EXT"], share=False, debug=True)
        # Load example 0
        

# Run the app
if __name__ == "__main__":
    app = App(config_path="cfg/config.ini")
    app.launch()