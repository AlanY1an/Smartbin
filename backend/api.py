from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import logging
from typing import Optional

from services.prediction_service import PredictionService
from services.waste_classification import WasteClassificationService

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.get("/")
async def root():
    return {"message": "Trash Classification API is running!"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form(None) 
):
    try:
        # Log all request information
        logger.info("Received prediction request")
        logger.info(f"Model name from form: {model_name}")
        
        # Check if model name is provided
        if model_name is None:
            available_models = [k for k, v in PredictionService.AVAILABLE_MODELS.items() if v is not None]
            error_msg = f"No model specified. Please choose a model from: {available_models}"
            logger.error(error_msg)
            return JSONResponse(
                status_code=400,
                content={
                    "status": 400,
                    "error": error_msg
                }
            )
        
        # Check if the requested model is available
        if not PredictionService.is_model_available(model_name):
            error_msg = f"Model '{model_name}' is not available. Please choose from: {[k for k, v in PredictionService.AVAILABLE_MODELS.items() if v is not None]}"
            logger.error(error_msg)
            return JSONResponse(
                status_code=400,
                content={
                    "status": 400,
                    "error": error_msg
                }
            )

        image = Image.open(BytesIO(await file.read())).convert("RGB")
        
        result_dict = PredictionService.predict_image(image, model_name)
        
        result_dict = WasteClassificationService.process_prediction_result(result_dict)
        
        logger.info(f"Successfully processed image with model {model_name}")
        
        return JSONResponse(
            content={
                "status": 200,
                "message": "Prediction successful",
                "model": model_name,
                **result_dict
            }
        )

    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.error(error_msg)
        return JSONResponse(
            status_code=400,
            content={
                "status": 400,
                "error": error_msg
            }
        )

# Add an endpoint to get available models
@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    available_models = [k for k, v in PredictionService.AVAILABLE_MODELS.items() if v is not None]
    return JSONResponse(
        content={
            "status": 200,
            "models": available_models
        }
    )

