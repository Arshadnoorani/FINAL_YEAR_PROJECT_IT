# image_processing.py - Image Preprocessing Utilities

from PIL import Image
import io

def preprocess_image(image_bytes: bytes) -> Image.Image:
    """
    Preprocesses the image input for model inference.
    
    Steps:
    - Load image from bytes.
    - Convert to RGB.
    - Resize to 224x224 (model standard input size).
    
    Args:
        image_bytes (bytes): Uploaded image in bytes format.

    Returns:
        PIL.Image.Image: Processed image ready for model input.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        return image
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")
