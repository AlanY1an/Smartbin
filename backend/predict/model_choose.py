import os
import logging
import torch
from typing import Optional, Dict, Callable

# Import various models
from model.DenseNet import densenet201
from model.EfficientNet import efficientnet_b7
from model.GoogLeNet import GoogLeNet
from model.LeNet import LeNet
from model.MobileNetV2 import mobilenet_v2
from model.MobileNetV3 import mobilenet_v3_large
from model.ResNet import resnet50
from model.AlexNet import AlexNet
from model.ShuffleNetV2 import shufflenet_v2_x2_0
from model.VGGNet import vgg
from model.RegNet import regnet_x_32gf


class TrainingConfig:
    """Training configuration class for managing basic model training parameters"""

    def __init__(
            self,
            model_name: str = 'ResNet',
            batch_size: int = 1,
            num_classes: int = 12,
            test_data_path: str = '../data/split-data/test',
            image_path: str = '',
            pretrained_weights: Optional[str] = None
    ):
        # Initialize training configuration parameters
        self.model_name = model_name
        self.batch_size = batch_size
        self.test_data_path = test_data_path
        self.num_classes = num_classes
        self.image_path = image_path
        self.pretrained_weights = pretrained_weights
        # Automatically detect and select the available device (GPU/CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Mac M1/M2/M3
        else:
            self.device = torch.device("cpu")  # CPU

class ModelChoose:
    """Model selection and initialization class, supporting multiple deep learning models and handling pretrained weights"""

    def __init__(self, config: TrainingConfig):
        """
        Initialize the model selector

        :param config: Training configuration object
        """
        self.config = config
        self.logger = self._setup_logger()
        # Define a mapping of supported models
        self.supported_models: Dict[str, Callable] = {
            "LeNet": self._create_lenet,
            "AlexNet": self._create_alexnet,
            "GoogLeNet": self._create_googlenet,
            "VGG": self._create_vgg19,
            "ResNet": self._create_resnet50,
            "RegNet": self._create_regnetx,
            "MobileNetV2": self._create_mobilenetv2,
            "MobileNetV3": self._create_mobilenetv3,
            "DenseNet": self._create_densenet201,
            "EfficientNet": self._create_efficientnet,
            "ShuffleNet": self._create_shufflenetv2,
        }

    def _create_model_factory(self, model_func: Callable) -> torch.nn.Module:
        """
        Generic model creation factory method

        :param model_func: Model creation function
        :return: Initialized model instance
        """
        return model_func(num_classes=self.config.num_classes).to(self.config.device)

    def _create_lenet(self) -> torch.nn.Module:
        return LeNet(num_classes=self.config.num_classes).to(self.config.device)

    def _create_alexnet(self) -> torch.nn.Module:
        return AlexNet(num_classes=self.config.num_classes, init_weights=True).to(self.config.device)

    def _create_googlenet(self) -> torch.nn.Module:
        return GoogLeNet(num_classes=self.config.num_classes).to(self.config.device)

    def _create_vgg19(self) -> torch.nn.Module:
        return vgg(model_name="vgg19", num_classes=self.config.num_classes, init_weights=True).to(self.config.device)

    def _create_resnet50(self) -> torch.nn.Module:
        return resnet50(num_classes=self.config.num_classes).to(self.config.device)

    def _create_regnetx(self) -> torch.nn.Module:
        return regnet_x_32gf(num_classes=self.config.num_classes).to(self.config.device)

    def _create_mobilenetv2(self) -> torch.nn.Module:
        return mobilenet_v2(num_classes=self.config.num_classes).to(self.config.device)

    def _create_mobilenetv3(self) -> torch.nn.Module:
        return mobilenet_v3_large(num_classes=self.config.num_classes).to(self.config.device)

    def _create_densenet201(self) -> torch.nn.Module:
        return densenet201(num_classes=self.config.num_classes).to(self.config.device)

    def _create_efficientnet(self) -> torch.nn.Module:
        return efficientnet_b7(num_classes=self.config.num_classes).to(self.config.device)

    def _create_shufflenetv2(self) -> torch.nn.Module:
        return shufflenet_v2_x2_0(num_classes=self.config.num_classes).to(self.config.device)

    def initialize_model(self) -> torch.nn.Module:
        """
        Initialize the model and load pretrained weights
        """
        # Retrieve the model creation function from the mapping dictionary
        model_creator = self.supported_models.get(
            self.config.model_name,
            self._create_resnet50  # Default to ResNet50
        )
        model = model_creator()
        model.load_state_dict(torch.load(self.config.pretrained_weights, weights_only=True))

        return model

    def _setup_logger(self):
        """
        Set up the logger

        :return: Configured logger
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)
