import os
import io
from flask import Flask, request, send_file
from flask_cors import CORS
import torch
from diffusers import StableDiffusionPipeline

# --- Model Configuration ---
# These paths will be relative to our app's location inside the container
BASE_MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
LORA_MODEL_PATH = "aesthetix-ai-lora-model-v1" # We will upload this folder to our deployment service
pipe = None

# --- Load the Model ---
# This runs once when the server starts
print("Loading model pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16)
pipe.load_lora_weights(LORA_MODEL_PATH)
pipe.to("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
print("✅ Model loaded successfully!")


# --- Flask App Definition ---
app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
    return "<h1>Aesthetix AI Backend is Live!</h1>"

@app.route("/redesign", methods=['POST'])
def redesign_image():
    global pipe
    if pipe is None:
        return "Error: Model is not loaded yet.", 503

    prompt = request.form.get('prompt', 'a photo of a room in modern style')
    print(f"Generating image with prompt: '{prompt}'")
    
    generated_image = pipe(prompt, num_inference_steps=50, guidance_scale=9).images[0]
    
    byte_io = io.BytesIO()
    generated_image.save(byte_io, 'PNG')
    byte_io.seek(0)
    
    print("✅ Image generated successfully. Sending response.")
    return send_file(byte_io, mimetype='image/png')

# The deployment service will run the app, so we don't need the `app.run()` block here.