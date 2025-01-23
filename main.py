from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow import keras
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import io
import os

import uvicorn

app = FastAPI()

app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
model = tf.keras.models.load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
   try:
       image_data = await file.read()
       image = Image.open(io.BytesIO(image_data)).convert("RGB")
       
       size = (224, 224)
       image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
       image_array = np.asarray(image)
       normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
       
       data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
       data[0] = normalized_image_array
       
       prediction = model.predict(data)
       index = np.argmax(prediction)
       class_name = class_names[index][2:].strip()
       confidence_score = float(prediction[0][index])
       
       return JSONResponse(content={
           "class": class_name,
           "confidence": confidence_score
       })
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
   with open("static/index.html") as f:
       html_content = f.read()
   return HTMLResponse(content=html_content, media_type="text/html")

if __name__ == "__main__":
   import uvicorn
   port = int(os.environ.get("PORT", 8000))
   uvicorn.run(app, host="0.0.0.0", port=port)