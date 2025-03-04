import logging
import json
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from predict.model_choose import ModelChoose, TrainingConfig


class Prediction:
    def __init__(self, config: TrainingConfig, model, logger=None):
        """
        Initializes the Prediction class.

        :param config: Training configuration.
        :param model: The trained model for prediction.
        :param logger: Logger instance for logging (optional).
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model = model
        self.classes = [
            'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
            'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
        ]
        if self.model is None:
            raise ValueError("Model is not initialized. Please provide a valid model instance.")

    @staticmethod
    def preprocess_image(image):
        """
        Preprocess the input image for model inference.

        :param image: Input image (PIL Image object).
        :return: Preprocessed image tensor.
        """
        try:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.1847, 0.1716, 0.1502],
                    std=[0.0678, 0.0615, 0.0552]
                )
            ])
            return transform(image).unsqueeze(0)  # Add batch dimension
        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {e}")

    def predict(self, image_tensor):
        """
        Performs prediction on the input image tensor.

        :param image_tensor: Preprocessed image tensor.
        :return: Predicted category and confidence score.
        """
        try:
            self.model.eval()
            with torch.no_grad():
                image_tensor = image_tensor.to(self.config.device)
                output = self.model(image_tensor)
                _, predicted_idx = torch.max(output, dim=1)
                confidence = torch.softmax(output, dim=1)[0, predicted_idx].item()
                return self.classes[predicted_idx.item()], confidence
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {e}")

    def run(self, image: Image.Image):
        """
        Runs the full prediction pipeline.

        :param image_path: Path to the image to be classified.
        :return: JSON result containing prediction details.
        """
        try:
            start_time = time.time()
            # Preprocess image
            image_tensor = self.preprocess_image(image)

            # Perform prediction
            prediction, confidence = self.predict(image_tensor)
            total_time = time.time() - start_time

            result = {
                "status": 200,
                "message": "Prediction successful",
                "model": self.config.model_name,
                "prediction": prediction,
                "confidence": f"{confidence * 100:.2f}%",
                "total_time": f"{total_time:.3f} seconds"
            }
            self.logger.info(f"Prediction completed: {result}")
            return json.dumps(result, ensure_ascii=False, indent=4)
        except Exception as e:
            error_response = {
                "status": 400,
                "message": f"Error: {str(e)}"
            }
            self.logger.error(f"Prediction failed: {error_response}")
            return json.dumps(error_response, ensure_ascii=False, indent=4)


def main():
    """
    Main function to test the Prediction class.
    """
    try:
        config = TrainingConfig(
            model_name='ResNet50',
            num_classes=12,
            image_path='../predict/test.jpg',
            pretrained_weights='../weights/ResNet/ResNet_model_0.00%.pth'
        )
        model_choose = ModelChoose(config)
        model = model_choose.initialize_model()

        predictor = Prediction(config, model, model_choose.logger)
        result = predictor.run(config.image_path)
        print(result)
    except Exception as e:
        print(f"Main function failed: {e}")


if __name__ == '__main__':
    main()
