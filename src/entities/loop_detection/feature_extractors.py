""" This module contains the feature extractors for the loop detection module.
    We implement NetVLAD feature extractor to compare it with ours.
    For NetVLAD we use the same hyper parameters as in CP-SLAM.
"""
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from src.entities.loop_detection.netvlad import NetVLAD


class BaseFeatureExtractor(object):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.weights_path = config["weights_path"]
        self.device = config.get("device", "cpu")

    def extract_features(self, image: Image) -> torch.Tensor:
        """ Extracts features from an image.
        Args:
            image: The input image.
        Returns:
            features: The extracted features.
        """
        raise NotImplementedError


class DINOFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.preprocess = AutoImageProcessor.from_pretrained(self.weights_path)
        self.model = AutoModel.from_pretrained(self.weights_path).to(self.device)
        self.embed_size = self.config.get("embed_size", 384)

    def extract_features(self, image: Image) -> torch.Tensor:
        """ Extracts DINOv2 features from an image. The features are normalized
            to be suitable for L2 distance search in the faiss database.
        Args:
            image: The input image.
        Returns:
            features: extracted DINOv2 features.
        """
        with torch.no_grad():
            inputs = self.preprocess(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
            features = features / features.norm(p=2, dim=1, keepdim=True)
            return features


class NetVLADFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model = NetVLAD({
            "checkpoint_path": self.weights_path,
            "whiten": True
        }).to(self.device)
        self.model.eval()
        self.embed_size = self.config.get("embed_size", 4096)

    def extract_features(self, image: Image) -> torch.Tensor:
        """ Extracts NetVLAD features from an image following CP-SLAM.
        Args:
            image: The input image.
        Returns:
            features: extracted NetVLAD features.
        """
        with torch.no_grad():
            return self.model(image)


def get_feature_extractor(config: dict) -> BaseFeatureExtractor:
    """ Returns the feature extractor based on the configuration.
    Args:
        config: The configuration dictionary.
    Returns:
        feature_extractor: The feature extractor object.
    """
    if config["feature_extractor_name"] == "dino":
        return DINOFeatureExtractor(config)
    elif config["feature_extractor_name"] == "netvlad":
        return NetVLADFeatureExtractor(config)
    else:
        raise NotImplementedError
