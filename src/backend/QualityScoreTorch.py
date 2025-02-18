import os
import cv2
import random
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import entropy
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel


class QualityScore:
    """Base class for quality score calculators like FID and KID."""
    def calculate(self, real_features, generated_features):
        pass


class FIDScore(QualityScore):
    """Class to calculate Frechet Inception Distance (FID)."""
    def calculate(self, real_features, generated_features):
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
    def calculate(self, real_features, generated_features):
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
        self.model = models.inception_v3(pretrained=True, transform_input=False)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def calculate(self, real_images=None, generated_images=None):
        """Calculate the Inception Score (IS) for generated images."""
        if generated_images is None:
            raise ValueError("Generated images are required for Inception Score calculation.")

        with torch.no_grad():
            # Preprocess and batch the images
            processed_images = torch.stack([self.transform(img) for img in generated_images])
            preds = torch.softmax(self.model(processed_images), dim=1).cpu().numpy()

        # Calculate the marginal distribution and KL divergence
        p_y = preds.mean(axis=0)
        scores = [entropy(pred, p_y) for pred in preds]
        inception_score = np.exp(np.mean(scores))

        return {"IS": inception_score}
    
    
class MMDScore(QualityScore):
    """Class to calculate Minimum Mean Discrepancy (MMD)."""
    def __init__(self, kernel='rbf', gamma=1.0):
        self.kernel = kernel
        self.gamma = gamma

    def calculate(self, real_features, generated_features):
        if self.kernel == 'rbf':
            k_rr = rbf_kernel(real_features, real_features, gamma=self.gamma)
            k_rg = rbf_kernel(real_features, generated_features, gamma=self.gamma)
            k_gg = rbf_kernel(generated_features, generated_features, gamma=self.gamma)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

        mmd = k_rr.mean() + k_gg.mean() - 2 * k_rg.mean()
        return {"MMD": mmd}
    
    
class FeatureExtractor:
    """Class to extract features using a pre-trained InceptionV3 model."""
    def __init__(self):
        self.model = models.inception_v3(pretrained=True, transform_input=False)
        self.model.fc = nn.Identity()  # Remove the fully connected layer
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
            "FID": FIDScore(),
            "KID": KIDScore(),
            "IS": InceptionScore(),
            "MMD": MMDScore(),

        }

    def load_images_from_directory(self, directory, img_size=(299, 299), max_images=5):
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
        real_features = self.feature_extractor.extract_features(real_images)
        generated_features = self.feature_extractor.extract_features(generated_images)

        results = {}
        for name, metric in self.metrics.items():
            if name == "IS":  # Inception Score does not use real images
                results.update(metric.calculate(None, generated_images))
            else:
                results.update(metric.calculate(real_features, generated_features))


        return results


if __name__ == "__main__":
    evaluator = QualityEvaluator()

    # Define datasets
    datasets = {
        "Real-Test": "data/quality_score/real-test",
        "Synthetic": "data/quality_score/synthetic",
        "New synthetic": "data/quality_score/new_synthetic/frames",
        "SVIA": "data/quality_score/svia"
    }
    #datasets = {
    #    "Real-Test": "data/quality_score/real-test",
    #    "Synthetic": "/media/daniel/TOSHIBA_EXT/BOSS_1/frames",
    #}
    reference_dataset = "data/quality_score/real"

    # Load reference and dataset images
    real_images = evaluator.load_images_from_directory(reference_dataset, max_images=99999)
    dataset_samples = {
        "Real": real_images,  # Add reference dataset
        **{name: evaluator.load_images_from_directory(path, max_images=10) for name, path in datasets.items()}
    }

    # Evaluate metrics (excluding Real vs Real comparison)
    results = {}
    for name, images in tqdm.tqdm(dataset_samples.items()):
        if name != "Real":
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
    output_file = "comparison_figure_with_metrics.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Figure saved to {output_file}")


## TO DO
# Añadir UMAP para gráfica visual 3D
# Rendered.AI features 4, 8, 16, 32, 64