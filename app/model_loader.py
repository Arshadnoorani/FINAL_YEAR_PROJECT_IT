# Load the model and processor for LLaVA
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig , AutoConfig
from peft import PeftModel
import torch
import os
from app.settings import settings

def load_model():
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor
    processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # Create offload directory (ensure it exists)
    offload_dir = "./offload"
    os.makedirs(offload_dir, exist_ok=True)

    # BitsAndBytes quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model_config = AutoConfig.from_pretrained("llava-hf/llava-1.5-7b-hf")
    # Load base model with quantization, device map, and offload folder
    base_model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        config=model_config,
        quantization_config=bnb_config,
        device_map={"": 0}, # Use CPU for initial loading
        offload_folder=offload_dir,
        torch_dtype=torch.float16
    )

    # Load PEFT fine-tuned model, using same device/offload setup
    model = PeftModel.from_pretrained(
        base_model,
        "mdnasif/LLaVA-med-MAKAUT",
        # device_map="auto",
        # offload_folder=offload_dir
    )

    model.eval()
    return model, processor

model, processor = load_model()
