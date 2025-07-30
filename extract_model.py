import os
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
from pdf2image import convert_from_path

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "model_artifacts"

processor = DonutProcessor.from_pretrained(model_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)

def extract_fields_from_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=200)
    if not images:
        return {"error": "No pages found."}
    image = images[0].convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    task_prompt = "<s>extract_fields</s>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id
    )
    sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return processor.token2json(sequence)
