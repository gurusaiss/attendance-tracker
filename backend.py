from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai, cv2, numpy as np
from PIL import Image
import io, os
from dotenv import load_dotenv

# Setup
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Extract names using Gemini
def extract_names_from_image(image_bytes):
    prompt = "You are an AI..."
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_bytes}])
    return [line.strip() for line in response.text.strip().split("\n") if line.strip()]

# Attendance logic
def detect_attendance(image_np, names, rows, cols):
    h, w, _ = image_np.shape
    grid_h, grid_w = h // rows, w // cols
    att = {}
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(names):
                tile = image_np[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                att[names[idx]] = 0.5 if np.mean(gray) < 40 or np.std(gray) < 15 else 1
    return att

@app.post("/process_attendance/")
async def process_attendance(file: UploadFile, rows: int = Form(...), cols: int = Form(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    names = extract_names_from_image(contents)
    if names[0].startswith("❌"):
        return JSONResponse(content={"status": "error", "message": names[0]})
    att_data = detect_attendance(image_cv, names, rows, cols)
    return JSONResponse(content={"status": "success", "data": att_data})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000)
