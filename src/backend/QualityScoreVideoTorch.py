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
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from pytorchvideo.models.hub import slowfast_r50  # Pretrained I3D alternative for FVD
from torch.nn import functional as F

# Import VBench (Requires pre-trained model)
try:
    import vbench
    use_vbench = True
except ImportError:
    print("VBench not installed. Skipping VBench evaluation.")
    use_vbench = False


class VideoQualityScore:
    """Base class for video quality score calculators like FVD and KVD."""
    def calculate(self, real_features, generated_features):
        pass


class FVDScore(VideoQualityScore):
    """Class to calculate Frechet Video Distance (FVD)."""
    def calculate(self, real_features, generated_features):
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fvd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return {"FVD": fvd}


class KVDScore(VideoQualityScore):
    """Class to calculate Kernel Video Distance (KVD)."""
    def calculate(self, real_features, generated_features):
        kvd_values = []
        n = min(len(real_features), len(generated_features))
        for _ in range(10):  # Use 10 subsets
            real_subset = real_features[np.random.choice(len(real_features), n, replace=False)]
            gen_subset = generated_features[np.random.choice(len(generated_features), n, replace=False)]
            k_rr = polynomial_kernel(real_subset, real_subset, degree=3)
            k_rg = polynomial_kernel(real_subset, gen_subset, degree=3)
            k_gg = polynomial_kernel(gen_subset, gen_subset, degree=3)
            kvd = k_rr.mean() + k_gg.mean() - 2 * k_rg.mean()
            kvd_values.append(kvd)
        return {"KVD": np.mean(kvd_values)}


class VideoInceptionScore(VideoQualityScore):
    """Class to calculate Inception Score (IS) for videos."""
    def __init__(self):
        self.model = slowfast_r50(pretrained=True)  # Pretrained I3D-like model
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize frames to match the model input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def calculate(self, real_videos=None, generated_videos=None):
        """Calculate the Inception Score (IS) for generated videos."""
        if generated_videos is None:
            raise ValueError("Generated videos are required for Video Inception Score calculation.")

        with torch.no_grad():
            processed_videos = torch.stack([self.transform(frame) for vid in generated_videos for frame in vid])
            preds = torch.softmax(self.model(processed_videos), dim=1).cpu().numpy()

        p_y = preds.mean(axis=0)
        scores = [entropy(pred, p_y) for pred in preds]
        inception_score = np.exp(np.mean(scores))

        return {"Video-IS": inception_score}


class VBenchScore(VideoQualityScore):
    """Class to calculate VBench score for videos (if VBench is installed)."""
    def calculate(self, real_videos, generated_videos):
        if not use_vbench:
            return {"VBench": None}

        # Assuming VBench provides a function to calculate similarity
        vbench_score = vbench.evaluate(real_videos, generated_videos)
        return {"VBench": vbench_score}


class VideoFeatureExtractor:
    """Class to extract video features using a pre-trained I3D model."""
    def __init__(self):
        self.model = slowfast_r50(pretrained=True)
        self.model.blocks[-1] = nn.Identity()  # Remove classification head
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, videos):
        with torch.no_grad():
            frames = torch.stack([self.transform(frame) for vid in videos for frame in vid])
            features = self.model(frames)
        return features.numpy()


class VideoQualityEvaluator:
    """Class to evaluate video similarity using quality scores."""
    def __init__(self):
        self.feature_extractor = VideoFeatureExtractor()
        self.metrics = {
            "FVD": FVDScore(),
            "KVD": KVDScore(),
            "Video-IS": VideoInceptionScore(),
            "VBench": VBenchScore() if use_vbench else None,
        }

    def load_videos_from_directory(self, directory, max_videos=5, frames_per_video=16):
        """Load videos from a directory using OpenCV."""
        videos = []
        files = np.random.choice(os.listdir(directory), min(max_videos, len(os.listdir(directory))), replace=False)
        for filename in files:
            vid_path = os.path.join(directory, filename)
            cap = cv2.VideoCapture(vid_path)
            frames = []
            while cap.isOpened() and len(frames) < frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            if len(frames) == frames_per_video:
                videos.append(frames)
        return videos

    def evaluate(self, real_videos, generated_videos):
        """Evaluate all video metrics."""
        real_features = self.feature_extractor.extract_features(real_videos)
        generated_features = self.feature_extractor.extract_features(generated_videos)

        results = {}
        for name, metric in self.metrics.items():
            if metric is not None:
                results.update(metric.calculate(real_videos if name == "VBench" else real_features, 
                                                generated_videos if name == "VBench" else generated_features))
        return results

if __name__ == "__main__":
    evaluator = VideoQualityEvaluator()

    # Define datasets
    video_datasets = {
        "Real-Test": "data/quality_score/real-test-videos",
        "Synthetic": "data/quality_score/synthetic-videos",
        "New Synthetic": "data/quality_score/new_synthetic_videos",
        "SVIA": "data/quality_score/svia_videos"
    }
    reference_video_dataset = "data/quality_score/real-videos"

    # Load reference and dataset videos
    real_videos = evaluator.load_videos_from_directory(reference_video_dataset, max_videos=10)
    dataset_samples = {
        "Real": real_videos,  # Add reference dataset
        **{name: evaluator.load_videos_from_directory(path, max_videos=5) for name, path in video_datasets.items()}
    }

    # Evaluate metrics (excluding Real vs Real comparison)
    results = {}
    for name, videos in tqdm.tqdm(dataset_samples.items()):
        if name != "Real":
            results[name] = evaluator.evaluate(real_videos, videos)

    # Dynamically extract metric names
    metric_names = list(next(iter(results.values())).keys())  # Get keys from the first result set

    # Create the figure with dynamic rows for metrics
    rows = len(metric_names) + 1  # Add one row for the image grid
    fig = plt.figure(figsize=(16, 6 * rows))
    gs = gridspec.GridSpec(rows, 1, height_ratios=[1] + [1] * len(metric_names))
    sns.set_theme(style="whitegrid")

    # Row 1: Single random sample frame from each dataset
    ax1 = plt.subplot(gs[0])
    ax1.axis("off")
    columns = len(dataset_samples)
    image_grid = gridspec.GridSpecFromSubplotSpec(
        1, columns, subplot_spec=gs[0], wspace=0.1
    )

    for col, (name, videos) in enumerate(dataset_samples.items()):
        random_frame = random.choice(videos)[0]  # Select the first frame from a random video
        ax = plt.subplot(image_grid[0, col])
        ax.imshow(random_frame)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.axis("off")

    # Rows 2+: Metrics
    for i, metric_name in enumerate(metric_names):
        image_grid = gridspec.GridSpecFromSubplotSpec(
            1, columns, subplot_spec=gs[i + 1], wspace=0.1
        )
        ax = plt.subplot(image_grid[0, 1:])
        metric_values = [results[name][metric_name] for name in video_datasets.keys()]
        sns.barplot(x=list(video_datasets.keys()), y=metric_values, ax=ax, palette="Set2", ci=None)
        ax.set_title(f"{metric_name} Comparison", fontsize=16)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_xlabel("Datasets", fontsize=12)

        # Annotate bars
        for j, value in enumerate(metric_values):
            ax.text(j, value + 0.01, f"{value:.2f}", ha='center', va='bottom', fontsize=10)

    # Save the plot
    output_file = "video_quality_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Figure saved to {output_file}")
