# config.py - Configuration settings for the FastAPI app

import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if exists)
load_dotenv()

class Settings:
    """Configuration settings for the the FastAPI app."""
    # MODEL_PATH = os.getenv("MODEL_PATH", "/model_file/llava_med_makaut.py")
    MODEL_PATH= os.getenv("MODEL_PATH", "mdnasif/LLaVA-med-MAKAUT")
    # MODEL_TYPE = os.getenv("MODEL_TYPE", "llava")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    API_TITLE = "LLaVA Medical Image Analysis API"
    API_VERSION = "1.0"

# Create an instance of Settings to be used in the application
settings = Settings()
