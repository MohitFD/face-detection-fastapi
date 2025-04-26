import os
import base64
import cv2
import numpy as np
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse

# from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import dlib
import face_recognition
from scipy.spatial import distance as dist
import pytesseract

import re

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set specific domains in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# For HTML template rendering
# templates = Jinja2Templates(directory="templates")

# Set current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Load models
liveness_model_path = os.path.join(current_directory, "mobilenetv2-epoch_10.hdf5")
liveness_model = load_model(liveness_model_path)

predictor_path = os.path.join(
    current_directory, "shape_predictor_68_face_landmarks.dat"
)
face_landmark_predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

# Constants
EAR_THRESHOLD = 0.2
EYE_AR_CONSEC_FRAMES = 1
blink_counter = 0
current_status = "FAILED"


# Request schema
class ImageData(BaseModel):
    image: str


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def predict_liveness(face_img):
    face_resized = cv2.resize(face_img, (224, 224))
    face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_resized = face_resized.astype("float32") / 255.0
    face_resized = np.expand_dims(face_resized, axis=0)
    pred = liveness_model.predict(face_resized)
    return pred[0] < 0.5  # Assume < 0.5 is fake


@app.post("/detect")
async def detect_and_mark(data: ImageData):
    global blink_counter, current_status

    frame = data.image
    if not frame:
        return JSONResponse(
            status_code=400,
            content={"status": False, "error": "No image data provided"},
        )

    try:
        image_data = base64.b64decode(frame)
    except Exception:
        return JSONResponse(
            status_code=400, content={"status": False, "error": "Invalid image data"}
        )

    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(
            status_code=400,
            content={"status": False, "error": "Could not decode image"},
        )

    face_locations = face_recognition.face_locations(img)
    if len(face_locations) == 0:
        return {"status": False, "error": "No face detected"}

    top, right, bottom, left = face_locations[0]
    face_img = img[top:bottom, left:right]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        face_img = img[y : y + h, x : x + w]

        if face_img.size == 0:
            continue

        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        landmarks = face_landmark_predictor(gray_face, dlib.rectangle(0, 0, w, h))

        left_eye = [
            (landmarks.part(i).x - x, landmarks.part(i).y - y) for i in range(36, 42)
        ]
        right_eye = [
            (landmarks.part(i).x - x, landmarks.part(i).y - y) for i in range(42, 48)
        ]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                print("Blink detected")
            blink_counter = 0

        is_real = predict_liveness(face_img)

        if is_real and blink_counter >= EYE_AR_CONSEC_FRAMES:
            current_status = "SUCCESS"
            print("Real human detected")
            return {"status": True, "message": current_status}
        else:
            current_status = "FAILED"
            return {"status": False, "message": current_status}

    return {"status": False, "message": current_status}

    #   IDPROOFEXTRACTOR


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)

# Patterns for each ID type
patterns = {
    "aadhar_number": r"\b\d{4}\s\d{4}\s\d{4}\b",
    "pan_number": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "drivng_license_number": r"\b[A-Z]{2}\d{2}\s?\d{11}\b",
    "voter_id_number": r"\b[A-Z]{3}[0-9]{7}\b",
    "passport_number": r"\b[A-Z][0-9]{7}\b",
    "passbook_number": r"\b\d{9,18}\b",
}


@app.post("/extracted-id")
async def extract_id(file: UploadFile = File(...), type: str = Form(...)):
    try:
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        text = pytesseract.image_to_string(gray)

        # Check if requested type exists in our patterns
        if type not in patterns:
            return JSONResponse(
                status_code=400, content={"message": f"Invalid type: {type}"}
            )

        if type == "account_number":
            for line in text.split("\n"):
                if "account" in line.lower() or "a/c" in line.lower():
                    match = re.findall(patterns[type], line)
                    if match:
                        return {"type": type, "value": match[0]}
        else:
            match = re.findall(patterns[type], text)
            if match:
                return {"type": type, "value": match[0]}

        return JSONResponse(
            status_code=404,
            content={"message": f"No valid {type.replace('_', ' ')} found!"},
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Run with: uvicorn filename:app --host 0.0.0.0 --port 5000
