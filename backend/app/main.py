from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import os

from app.models.predictor import SkinDiseasePredictor
from app.routes import predict

# Global predictor instance
predictor_instance = None


def _resolve_model_path() -> str:
    """Local path, or download from Hugging Face Hub if HF_MODEL_REPO_ID is set."""
    model_path = os.getenv("MODEL_PATH", "trained_model/SKIN_MODEL_BEST.keras")
    repo_id = os.getenv("HF_MODEL_REPO_ID")
    if repo_id and not os.path.isfile(model_path):
        from huggingface_hub import hf_hub_download

        filename = os.getenv("HF_MODEL_FILENAME", "SKIN_MODEL_BEST.keras")
        out_dir = os.path.dirname(model_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=out_dir,
        )
        return downloaded
    return model_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    global predictor_instance
    model_path = _resolve_model_path()
    
    # Try different possible paths
    possible_paths = [
        model_path,
        'trained_model/SKIN_MODEL_BEST.keras',
        '../trained_model/SKIN_MODEL_BEST.keras',
        './trained_model/SKIN_MODEL_BEST.keras'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    print(f"Loading model from: {model_path}")
    predictor_instance = SkinDiseasePredictor(model_path)
    
    # Set predictor in routes
    predict.set_predictor(predictor_instance)
    
    yield
    
    # Shutdown: Clean up
    print("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Skin Disease Prediction API",
    description="API for skin disease classification using deep learning",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routes
app.include_router(predict.router)

@app.get("/info")
async def get_info():
    """Get model information"""
    if predictor_instance and predictor_instance.is_model_loaded():
        return {
            "model_loaded": True,
            "classes": predictor_instance.class_names,
            "image_size": predictor_instance.img_size
        }
    else:
        return {
            "model_loaded": False,
            "error": "Model not loaded"
        }