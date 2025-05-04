from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
import io
import torch

from app.model_loader import model, processor

router = APIRouter()

def process_image(image: UploadFile):
    """Converts uploaded image to PIL format."""
    try:
        image_data = Image.open(io.BytesIO(image.file.read())).convert("RGB")
        return image_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def improve_prompt(text: str) -> str:
    """Improves the user prompt if it's too short or vague."""
    if len(text.strip()) < 10:
        # Auto-extend very short prompts
        text = "Describe the visible medical condition in detail."
    elif "Alzheimer" in text or "alzheimers" in text.lower():
        # If question related to Alzheimer's
        text = f"Analyze the image for signs of Alzheimer's disease. {text}"
    return text

@router.post("/predict")
async def predict(
    image: UploadFile = File(...),
    text: str = "Describe the medical condition in this image"
):
    """Accepts image and optional prompt text, returns LLaVA response."""
    try:
        # Load and preprocess image
        image_data = process_image(image)

        # Improve user prompt
        improved_text = improve_prompt(text)

        # Build final prompt with <image> token
        prompt = f"<image>\n{improved_text}"

        # Tokenize input
        inputs = processor(
            text=prompt,
            images=[image_data],  # LLaVA expects a list of images
            return_tensors="pt"
        )

        # Move inputs to model device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.base_model.model.language_model.model.embed_tokens.weight.device)

        # Generate prediction
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7
            )

        # Decode and clean up the output
        response_text = processor.tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # If model repeats back the input (lazy generation), handle it
        if response_text.lower().strip() == improved_text.lower().strip():
            response_text = "The model could not confidently describe the image based on the provided prompt."

        return {"response": response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
