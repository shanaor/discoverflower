from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
import io
import os

import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model and labels when the application starts
try:
    model = keras.models.load_model("keras_Model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
except Exception as e:
    print(f"Error loading model or labels: {e}")
    model = None
    class_names = None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model or not class_names:
        raise HTTPException(status_code=500, detail="Model or labels not loaded")
    
    try:
        # Read and process the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Resize and preprocess the image
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # Prepare data for prediction
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # Make prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index][2:].strip()  # Remove the "1." prefix and strip whitespace
        confidence_score = float(prediction[0][index])
        
        return JSONResponse(content={
            "class": class_name,
            "confidence": confidence_score
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve the HTML page
@app.get("/")
async def read_root():
    with open("static/index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, media_type="text/html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)