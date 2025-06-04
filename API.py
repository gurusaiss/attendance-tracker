# backend.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

def extract_names_from_image(image_bytes):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = """
You are an AI helping to detect student names in a classroom screenshot (Google Meet or Zoom grid). 
Extract all names **exactly as they appear** on screen, in order (left to right, top to bottom). 
List only the names, one per line. No other text.
"""
    try:
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])
        return [line.strip() for line in response.text.strip().split("\n") if line.strip()]
    except Exception as e:
        return [f"❌ Error: {str(e)}"]

def detect_attendance(image_np, names, rows, cols):
    h, w, _ = image_np.shape
    grid_h, grid_w = h // rows, w // cols
    attendance = {}

    for i in range(rows):
        for j in range(cols):
            cell_index = i * cols + j
            if cell_index < len(names):
                name = names[cell_index]
                x1, y1 = j * grid_w, i * grid_h
                x2, y2 = x1 + grid_w, y1 + grid_h
                cell = image_np[y1:y2, x1:x2]

                gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                variation = np.std(gray)

                if brightness < 40 or variation < 15:
                    attendance[name] = 0.5
                else:
                    attendance[name] = 1
            else:
                break

    for name in names:
        if name not in attendance:
            attendance[name] = 0

    return attendance

@app.post("/process_attendance/")
async def process_attendance(file: UploadFile, rows: int = Form(...), cols: int = Form(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    names = extract_names_from_image(contents)
    if names and not names[0].startswith("❌"):
        attendance_data = detect_attendance(image_cv, names, rows, cols)
        return JSONResponse(content={"status": "success", "data": attendance_data})
    else:
        return JSONResponse(content={"status": "error", "message": names[0] if names else "Unknown error"})

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000)
