import os
import sys
sys.path.append(os.path.abspath('.'))

import tqdm
import shutil

from src.backend.SpermatozoonFactory import SpermatozoonFactory
from src.backend.DebrisFactory import DebrisFactory
from src.backend.BackgroundGenerator import BackgroundGenerator
from src.backend.ConfigParser import SequenceConfig
from src.backend.SequenceGenerator import SequenceGenerator
from src.backend.utils import save_gif


class DatasetMaker:
    def __init__(self, num_sequences: int, num_frames: int, sequence_config, sf, df, bgg, output_dir="dataset", dpi=300, image_augmentor=None):
        """
        Initialize the DatasetMaker.

        Parameters:
            num_sequences (int): The number of sequences to generate.
            sequence_length (int): Number of frames per sequence.
            sequence_config: Configuration object for sequences.
            sf: SpermatozoonFactory instance.
            df: DebrisFactory instance.
            bgg: BackgroundGenerator instance.
            output_dir (str): Directory to save the dataset.
            dpi (int): Dots per inch for generated images.
        """
        self.num_sequences = int(num_sequences)
        self.num_frames = int(num_frames)
        self.sequence_config = sequence_config
        self.sf = sf
        self.df = df
        self.bgg = bgg
        self.output_dir = output_dir
        self.dpi = dpi
        self.image_augmentor = image_augmentor
        self.sg = SequenceGenerator(self.num_frames, self.sequence_config, self.sf, self.df, self.bgg, self.dpi, self.image_augmentor)

        os.makedirs(self.output_dir, exist_ok=True)

    def generate_sequence(self, n_seq, remove_old=True):
        """
        Generate a single sequence.
        """
        sequence_dir = os.path.join(self.output_dir, "frames")
        gif_dir = os.path.join(self.output_dir, "gifs")
        if remove_old:
            if os.path.isdir(sequence_dir):
                shutil.rmtree(sequence_dir)
            if os.path.isdir(gif_dir):
                shutil.rmtree(gif_dir)
        os.makedirs(gif_dir, exist_ok=True)
        
        self.sg.generate_sequence(sequence_dir)
        save_gif(sequence_dir, n_seq=n_seq, save_path=os.path.join(gif_dir, f"{n_seq:06d}.gif"), duration=100, loop=0)


    def generate_dataset(self):
        """
        Generate the entire dataset with multiprocessing.
        """
        progress = tqdm.tqdm(total=self.num_sequences, desc="Generating dataset", leave=False, ascii="▱▰", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        for n in range(self.num_sequences):
            self.generate_sequence(n)
            progress.update(1)
            yield [os.path.join(self.output_dir, "gifs", f"{n:06d}.gif") for p in os.listdir(os.path.join(self.output_dir, "gifs"))], ">" + progress.desc + " " + progress.format_meter(
                                                            n=n+1,
                                                            total=self.num_sequences,
                                                            elapsed=progress.format_dict['elapsed'],
                                                            ascii="▱▰",
                                                            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
                                                        ) + "<br/><br/>"
        progress.close()
        

if __name__ == "__main__":
    # Example usage
    config_path = os.path.join(os.getcwd(), 'cfg', 'config.ini')
    sequence_config = SequenceConfig(config_path)
    
    sf = SpermatozoonFactory(species_dict_path="cfg/species/sampleSpecie.json", style_config_path="cfg/styles/base.json")
    df = DebrisFactory(debris_dict_path="cfg/debris.json", style_config_path="cfg/styles/base.json")
    
    paths = "/home/daniel/Documents/Projects/Kubus/Morfología/Data/vids_processed"
    image_paths = os.listdir(paths)
    image_paths = [os.path.join(paths, p) for p in image_paths]
    bgg = BackgroundGenerator()
    bgg.setGenerationMethod('list', paths=image_paths)

    dm = DatasetMaker(num_sequences=10, num_frames=100, sequence_config=sequence_config, sf=sf, df=df, bgg=bgg, output_dir="dataset")
    dm.generate_dataset()
