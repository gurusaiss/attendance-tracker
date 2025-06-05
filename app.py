# Load dependencies
import streamlit as st, pandas as pd, cv2, numpy as np
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# App UI setup
st.set_page_config(page_title="Zoom/Meet Attendance Tracker", layout="wide")
st.title("📸 Zoom/Meet Attendance Tracker")

# Upload Image
screenshot = st.file_uploader("Upload Screenshot", type=["png", "jpg"])
cols = st.number_input("Columns", min_value=1, max_value=15, value=5)
rows = st.number_input("Rows", min_value=1, max_value=10, value=3)
if screenshot:
    image = Image.open(screenshot)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image)

# OCR + Attendance Logic
def extract_names_from_full_image(image_bytes):
    prompt = """You are an AI... (name-only output)"""
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content([prompt, {"mime_type": "image/jpeg", "data": image_bytes}])
    return [line.strip() for line in response.text.strip().split("\n") if line.strip()]

def detect_attendance_with_global_names(image, names, rows, cols):
    h, w, _ = image.shape
    grid_h, grid_w = h // rows, w // cols
    att = {}
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(names):
                tile = image[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                variation = np.std(gray)
                att[names[idx]] = 0.5 if brightness < 40 or variation < 15 else 1
    return att

if st.button("📍 Detect Attendance") and screenshot:
    names = extract_names_from_full_image(screenshot.getvalue())
    if names[0].startswith("❌"):
        st.error("Gemini API failed")
    else:
        result = detect_attendance_with_global_names(image_cv, names, int(rows), int(cols))
        df = pd.DataFrame(result.items(), columns=["Name", "Attendance Score"])
        st.success("✅ Done!")
        st.dataframe(df)
        st.download_button("📥 Download CSV", df.to_csv(index=False), "attendance.csv")
 