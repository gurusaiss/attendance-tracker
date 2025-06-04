import streamlit as st
import pandas as pd
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set Gemini API key securely
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# App title
st.set_page_config(page_title="Zoom/Meet Attendance Tracker", layout="wide")
st.title("📸 Zoom/Meet Attendance Tracker")

# Upload section
screenshot = st.file_uploader("Upload Screenshot of Zoom/Meet", type=["png", "jpg", "jpeg"])
cols = st.number_input("Enter No. of Columns (People per Row)", min_value=1, max_value=15, value=5)
rows = st.number_input("Enter No. of Rows", min_value=1, max_value=10, value=3)

if screenshot:
    image = Image.open(screenshot)
    st.image(image, caption="Uploaded Screenshot", use_column_width=True)

    # Resize image for faster processing
    image_resized = image.copy()
    image_resized.thumbnail((800, 800))
    image_cv = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)

def extract_names_from_full_image(image_bytes):
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

def detect_attendance_with_global_names(image, names, rows, cols):
    h, w, _ = image.shape
    grid_h, grid_w = h // rows, w // cols
    attendance = {}

    for i in range(rows):
        for j in range(cols):
            cell_index = i * cols + j
            if cell_index < len(names):
                name = names[cell_index]
                x1, y1 = j * grid_w, i * grid_h
                x2, y2 = x1 + grid_w, y1 + grid_h
                cell = image[y1:y2, x1:x2]

                gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                variation = np.std(gray)

                if brightness < 40 or variation < 15:
                    attendance[name] = 0.5  # Likely video off
                else:
                    attendance[name] = 1  # Video on
            else:
                break

    for name in names:
        if name not in attendance:
            attendance[name] = 0

    return attendance

if st.button("📍 Detect Attendance"):
    if screenshot:
        with st.spinner("🧠 Extracting names with Gemini..."):
            resized_img = image.copy()
            resized_img.thumbnail((800, 800))
            buffer = BytesIO()
            resized_img.save(buffer, format="JPEG", quality=60)
            image_bytes = buffer.getvalue()
            all_names = extract_names_from_full_image(image_bytes)

        if all_names and not all_names[0].startswith("❌ Error"):
            st.success("✅ Names Extracted")
            st.write("Detected Names:", all_names)

            with st.spinner("🧑‍💻 Detecting attendance from video tiles..."):
                att = detect_attendance_with_global_names(image_cv, all_names, int(rows), int(cols))
                df = pd.DataFrame(att.items(), columns=["Name", "Attendance Score"])
                st.success("✅ Attendance Processed!")
                st.dataframe(df)
                st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="attendance.csv")
        else:
            st.error(f"Error extracting names: {all_names[0] if all_names else 'Unknown error'}")
    else:
        st.warning("Please upload a screenshot image first.")
