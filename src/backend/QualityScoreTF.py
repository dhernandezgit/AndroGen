import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from scipy.stats import entropy
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import polynomial_kernel
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.image import resize
from abc import ABC, abstractmethod
from typing import List

class QualityScore(ABC):
    """Abstract base class for quality score calculators like FID, KID, and IS."""
    @abstractmethod
    def calculate(self, real_images: np.ndarray, generated_images: np.ndarray) -> dict:
        pass


class FIDScore(QualityScore):
    """Class to calculate Frechet Inception Distance (FID)."""
    def calculate(self, real_features: np.ndarray, generated_features: np.ndarray) -> dict:
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return {"FID": fid}


class KIDScore(QualityScore):
    """Class to calculate Kernel Inception Distance (KID)."""
    def calculate(self, real_features: np.ndarray, generated_features: np.ndarray) -> dict:
        kid_values = []
        n = min(len(real_features), len(generated_features))
        for _ in range(10):  # Use 10 subsets
            real_subset = real_features[np.random.choice(len(real_features), n, replace=False)]
            gen_subset = generated_features[np.random.choice(len(generated_features), n, replace=False)]
            k_rr = polynomial_kernel(real_subset, real_subset, degree=3)
            k_rg = polynomial_kernel(real_subset, gen_subset, degree=3)
            k_gg = polynomial_kernel(gen_subset, gen_subset, degree=3)
            kid = k_rr.mean() + k_gg.mean() - 2 * k_rg.mean()
            kid_values.append(kid)
        return {"KID": np.mean(kid_values)}


class InceptionScore(QualityScore):
    """Class to calculate Inception Score (IS)."""
    def __init__(self):
        self.inception_model = InceptionV3(include_top=True, weights='imagenet')

    def calculate(self, real_images: np.ndarray, generated_images: np.ndarray) -> dict:
        """Calculate the Inception Score (IS)."""
        generated_images = preprocess_input(np.array(generated_images))
        preds = self.inception_model.predict(generated_images, batch_size=32)
        p_y = np.mean(preds, axis=0)
        scores = [entropy(pred, p_y) for pred in preds]
        is_score = np.exp(np.mean(scores))
        return {"IS": is_score}


class FeatureExtractor:
    """Class to extract features using a pre-trained InceptionV3 model."""
    def __init__(self):
        self.model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        images = preprocess_input(np.array(images))
        return self.model.predict(images, batch_size=32)


class QualityEvaluator:
    """High-level class to evaluate distribution similarity using quality scores like FID, KID, and IS."""
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.metrics = {
            "FID": FIDScore(),
            "KID": KIDScore(),
            "IS": InceptionScore()
        }

    def load_images_from_directory(self, directory, img_size=(299, 299), max_images=10):
        """Load images from a directory using OpenCV."""
        images = []
        for idx, filename in enumerate(os.listdir(directory)):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                images.append(img)
            if idx == max_images:
                break
        return images

    def evaluate(self, real_images: list, generated_images: list) -> dict:
        """Evaluate all metrics and return a consolidated dictionary."""
        real_features = self.feature_extractor.extract_features(real_images)
        generated_features = self.feature_extractor.extract_features(generated_images)

        results = {}
        for name, metric in self.metrics.items():
            if name == "IS":
                results.update(metric.calculate(None, generated_images))
            else:
                results.update(metric.calculate(real_features, generated_features))

        return results
    
# Main script
if __name__ == "__main__":
    evaluator = QualityEvaluator()

    # Define datasets
    #datasets = {
    #    "Real-Test": "data/quality_score/real-test",
    #    "Synthetic": "data/quality_score/synthetic",
    #    "SVIA": "data/quality_score/svia"
    #}
    # Define datasets
    ref = "BOSS-Ref"
    datasets = {
        "BOSS-Test": "/media/daniel/TOSHIBA_EXT/DatasetsMetrics/BOSS/set2",
        "BOSS-Synthetic": "/media/daniel/TOSHIBA_EXT/SBOSSV0/set1",
    }
    reference_dataset = "/media/daniel/TOSHIBA_EXT/DatasetsMetrics/BOSS/set1"

    #Load reference and dataset images
    real_images = evaluator.load_images_from_directory(reference_dataset)
    dataset_samples = {
        ref: real_images,  # Add reference dataset
        **{name: evaluator.load_images_from_directory(path) for name, path in datasets.items()}
    }

    # Evaluate metrics (excluding Real vs Real comparison)
    results = {}
    for name, images in dataset_samples.items():
        if name != ref:
            results[name] = evaluator.evaluate(real_images, images)

    # Dynamically extract metric names
    metric_names = list(next(iter(results.values())).keys())  # Get keys from the first result set

    # Create the figure with dynamic rows for metrics
    rows = len(metric_names) + 1  # Add one row for the image grid
    fig = plt.figure(figsize=(16, 6 * rows))
    gs = gridspec.GridSpec(rows, 1, height_ratios=[1] + [1] * len(metric_names))
    sns.set_theme(style="whitegrid")

    # Row 1: Single random sample image for each dataset
    ax1 = plt.subplot(gs[0])
    ax1.axis("off")
    columns = len(dataset_samples)
    image_grid = gridspec.GridSpecFromSubplotSpec(
        1, columns, subplot_spec=gs[0], wspace=0.1
    )

    for col, (name, images) in enumerate(dataset_samples.items()):
        random_img = random.choice(images)  # Select a single random image
        ax = plt.subplot(image_grid[0, col])
        ax.imshow(random_img)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.axis("off")

    # Rows 2+: Metrics
    for i, metric_name in enumerate(metric_names):
        image_grid = gridspec.GridSpecFromSubplotSpec(
        1, columns, subplot_spec=gs[i + 1], wspace=0.1
        )
        ax = plt.subplot(image_grid[0, 1:])
        metric_values = [results[name][metric_name] for name in datasets.keys()]
        sns.barplot(x=list(datasets.keys()), y=metric_values, ax=ax, palette="Set2", ci=None)
        ax.set_title(f"{metric_name} Comparison", fontsize=16)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_xlabel("Datasets", fontsize=12)

        # Annotate bars
        for j, value in enumerate(metric_values):
            ax.text(j, value + 0.01, f"{value:.2f}", ha='center', va='bottom', fontsize=10)

    # Save the plot
    output_file = "BOSS_comparison_figure_with_metrics.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Figure saved to {output_file}")


## TO DO
# Añadir UMAP para gráfica visual 3D
# Rendered.AI features 4, 8, 16, 32, 64