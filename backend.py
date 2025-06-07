from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai, cv2, numpy as np
from PIL import Image
import io, os, tempfile
from datetime import timedelta

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Existing image attendance functions here (extract_names_from_image, detect_attendance)

def extract_names_from_frame(frame):
    is_success, buffer = cv2.imencode(".jpg", frame)
    if not is_success:
        return []
    img_bytes = buffer.tobytes()
    prompt = """
    You are an AI helping to detect student names in a classroom screenshot (Google Meet or Zoom grid). 
    Extract all names exactly as they appear on screen, in order (left to right, top to bottom). 
    List only the names, one per line. No other text.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": img_bytes}])
        return [line.strip() for line in response.text.strip().split("\n") if line.strip()]
    except Exception as e:
        return []

def detect_changes_over_time(name_snapshots, timestamps):
    logs = []
    previous_names = set()
    for i, current_names in enumerate(name_snapshots):
        current_set = set(current_names)
        time = timestamps[i]
        joined = current_set - previous_names
        left = previous_names - current_set
        for name in joined:
            logs.append({"Timestamp": time, "Change": f"{name} joined"})
        for name in left:
            logs.append({"Timestamp": time, "Change": f"{name} left"})
        previous_names = current_set
    return logs

@app.post("/process_video_attendance/")
async def process_video_attendance(file: UploadFile, interval_minutes: int = Form(10)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(contents)
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * interval_minutes * 60)

    frames = []
    timestamps = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval_frames == 0:
            small_frame = cv2.resize(frame, (640, 360))
            timestamp = str(timedelta(seconds=int(count / fps)))
            frames.append(small_frame)
            timestamps.append(timestamp)
        count += 1
    cap.release()
    os.unlink(video_path)

    name_snapshots = []
    for frame in frames:
        names = extract_names_from_frame(frame)
        name_snapshots.append(names)

    event_logs = detect_changes_over_time(name_snapshots, timestamps)

    return JSONResponse(content={
        "status": "success",
        "name_snapshots": [{"timestamp": t, "names": names} for t, names in zip(timestamps, name_snapshots)],
        "event_logs": event_logs
    })

# Keep your existing /process_attendance endpoint as is
