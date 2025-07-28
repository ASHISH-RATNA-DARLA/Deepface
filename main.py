from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import numpy as np
import cv2

app = FastAPI()

# Enable requests from any origin (for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        result = result[0] if isinstance(result, list) else result
        return {
            "dominant_emotion": result["dominant_emotion"],
            "emotions": result["emotion"]
        }
    except Exception as e:
        return {"error": str(e)}
