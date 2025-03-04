from typing import Dict
from PIL import Image
import json
from predict.model_choose import ModelChoose, TrainingConfig
from predict.prediction import Prediction

class PredictionService:
    # Available models mapping
    AVAILABLE_MODELS = {
        "AlexNet": "./weights/AlexNet/AlexNet_model_92.04%.pth",
        "DenseNet": "./weights/DenseNet/DenseNet_model_68.61%.pth",
        "EfficientNet": "./weights/EfficientNet/EfficientNet_model_91.46%.pth",
        "GoogLeNet": "./weights/GoogLeNet/GoogLeNet_model_0.00%.pth",
        "LeNet": "./weights/LeNet/LeNet_model_53.46%.pth",
        "MobileNetV2": "./weights/MobileNetV2/MobileNetV2_model_95.34%.pth",
        "MobileNetV3": "./weights/MobileNetV3/MobileNetV3_model_94.95%.pth",
        "RegNet": "./weights/RegNet/RegNet_model_97.41%.pth",
        "ResNet": "./weights/ResNet/ResNet_model_98.14%.pth",
        "ShuffleNetV2": "./weights/ShuffleNetV2/ShuffleNetV2_model_95.40%.pth",
        "VGG": "./weights/VGG/VGG_model_94.69%.pth",
        "VGGNet": None
    }


    @staticmethod
    def is_model_available(model_name: str) -> bool:
        """
        Check if the requested model is available and has weights file
        """
        return model_name in PredictionService.AVAILABLE_MODELS and PredictionService.AVAILABLE_MODELS[model_name] is not None

    @staticmethod
    def predict_image(image: Image.Image, model_name: str = "ResNet") -> Dict:
        """
        Predict the class of an image using the specified model
        Returns:
            Dict: A dictionary containing prediction results
        """
        # Load the selected model
        config = TrainingConfig(
            model_name=model_name,
            num_classes=12,
            pretrained_weights=PredictionService.AVAILABLE_MODELS[model_name]
        )
        model_choose = ModelChoose(config)
        model = model_choose.initialize_model()

        # Initialize Prediction with the selected model
        predictor = Prediction(config, model)

        # Process image and get result
        result = predictor.run(image)  # This returns a JSON string
        return json.loads(result)  # Convert JSON string to Dict 