import os
import cv2
import random
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
import clip
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from PIL import Image
import umap.umap_ as umap
from mpl_toolkits.mplot3d import Axes3D

class QualityScore:
    """Base class for quality score calculators like FID and KID."""
    def calculate(self, real_images, generated_images):
        pass

class FIDScore(QualityScore):
    """Class to calculate Frechet Inception Distance (FID) using torchmetrics."""
    def __init__(self, feature=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.fid = FrechetInceptionDistance(feature=feature).to(device)
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).byte())  # Ensure uint8 format
        ])

    def calculate(self, real_images, generated_images):
        """Compute FID using a batch size of 1, iterating through all images."""
        for real_img, gen_img in zip(real_images, generated_images):
            real_img = self.transform(real_img).unsqueeze(0).to(self.device)  # [1, C, H, W]
            gen_img = self.transform(gen_img).unsqueeze(0).to(self.device)

            self.fid.update(real_img, real=True)
            self.fid.update(gen_img, real=False)

        fid_score = self.fid.compute().item()
        self.fid.reset()
        return {"FID": fid_score}

class KIDScore(QualityScore):
    """Class to calculate Kernel Inception Distance (KID) using torchmetrics."""
    def __init__(self, feature=2048, subsample_size=1000, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.feature = feature
        self.subsample_size = subsample_size  # Default value
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).byte())  # Ensure uint8 format
        ])

    def calculate(self, real_images, generated_images):
        """Compute KID using a batch size of 1, iterating through all images."""
        # Adjust subset_size dynamically
        min_size = min(len(real_images), len(generated_images))
        subset_size = min(self.subsample_size, min_size)  # Prevent exceeding available samples
        
        self.kid = KernelInceptionDistance(feature=self.feature, subset_size=subset_size).to(self.device)

        for real_img, gen_img in zip(real_images, generated_images):
            real_img = self.transform(real_img).unsqueeze(0).to(self.device)  # [1, C, H, W]
            gen_img = self.transform(gen_img).unsqueeze(0).to(self.device)

            self.kid.update(real_img, real=True)
            self.kid.update(gen_img, real=False)

        kid_mean, kid_std = self.kid.compute()
        kid_score = kid_mean.item()

        self.kid.reset()
        return {"KID": kid_score}


class FeatureExtractor:
    """Class to extract features using a pre-trained InceptionV3 model."""
    def __init__(self):
        self.model = models.inception_v3(pretrained=True, transform_input=False)
        self.model.fc = nn.Identity()
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, images):
        with torch.no_grad():
            images = torch.stack([self.transform(img) for img in images])
            features = self.model(images)
        return features.numpy()

class QualityEvaluator:
    """High-level class to evaluate distribution similarity using quality scores."""
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.metrics = {
            "KID": KIDScore(),
            "FID": FIDScore(),
        }

    def load_images_from_directory(self, directory, img_size=(299, 299), max_images=350):
        """Load images from a directory using OpenCV."""
        images = []
        files = np.random.choice(os.listdir(directory), min(max_images, len(os.listdir(directory))), replace=False)
        for idx, filename in enumerate(files):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                images.append(img)
            if idx == max_images:
                break
        return images

    def evaluate(self, real_images, generated_images):
        """Evaluate all metrics and return a consolidated dictionary."""
        results = {}
        for name, metric in self.metrics.items():
            results.update(metric.calculate(real_images, generated_images))
        return results

if __name__ == "__main__":
    evaluator = QualityEvaluator()

    dataset = "VISEM"
    ref = f"{dataset}-Ref"
    datasets = {
        f"{dataset}-Test": f"/media/daniel/TOSHIBA_EXT/DatasetsMetrics/{dataset}/set2",
        f"{dataset}-Synthetic": f"/media/daniel/TOSHIBA_EXT/S{dataset}V3/set2",
        "BOSS": "/media/daniel/TOSHIBA_EXT/DatasetsMetrics/BOSS/set2",
        "SVIA": "/media/daniel/TOSHIBA_EXT/DatasetsMetrics/SVIA/set2",
    }
    reference_dataset = f"/media/daniel/TOSHIBA_EXT/DatasetsMetrics/{dataset}/set1"

    real_images = evaluator.load_images_from_directory(reference_dataset, max_images=350)
    dataset_samples = {
        ref: real_images,
        **{name: evaluator.load_images_from_directory(path, max_images=350) for name, path in datasets.items()}
    }

    results = {}
    for name, images in tqdm.tqdm(dataset_samples.items()):
        if name != ref:
            results[name] = evaluator.evaluate(real_images, images)

    metric_names = list(next(iter(results.values())).keys())

    fig = plt.figure(figsize=(16, 6 * len(metric_names)))
    gs = gridspec.GridSpec(len(metric_names), 1)

    sns.set_theme(style="whitegrid")

    for i, metric_name in enumerate(metric_names):
        ax = plt.subplot(gs[i])
        metric_values = [results[name][metric_name] for name in datasets.keys()]
        sns.barplot(x=list(datasets.keys()), y=metric_values, ax=ax, palette="Set2", ci=None)
        ax.set_title(f"{metric_name} Comparison", fontsize=16)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_xlabel("Datasets", fontsize=12)

        for j, value in enumerate(metric_values):
            ax.text(j, value + 0.01, f"{value:.2f}", ha='center', va='bottom', fontsize=10)

    output_file = f"{ref.split('-')[0]}V3_comparison_figure_with_metrics.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Figure saved to {output_file}")
