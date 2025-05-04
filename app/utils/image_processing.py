# image_processing.py - Image Preprocessing Utilities

from PIL import Image

def preprocess_image(image: Image):
    """Resizes and normalizes an image for model input."""
    try:
        image = image.resize((224, 224)).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")
