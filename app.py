import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import timedelta

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Zoom/Meet Attendance Tracker", layout="wide")
st.title("🧑‍🏫 Zoom/Meet Attendance Tracker")

tab1, tab2 = st.tabs(["📸 Screenshot Attendance", "🎥 Video Attendance"])

######################################
# 📸 TAB 1 — Screenshot Attendance
######################################

with tab1:
    screenshot = st.file_uploader("Upload Screenshot of Zoom/Meet", type=["png", "jpg", "jpeg"])
    cols = st.number_input("Enter No. of Columns (People per Row)", min_value=1, max_value=15, value=5)
    rows = st.number_input("Enter No. of Rows", min_value=1, max_value=10, value=3)

    if screenshot:
        image = Image.open(screenshot)
        st.image(image, caption="Uploaded Screenshot", use_column_width=True)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

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

                    # Simple heuristics for attendance presence
                    if brightness < 40 or variation < 15:
                        attendance[name] = 0.5  # Possibly video off or inactive
                    else:
                        attendance[name] = 1  # Present
                else:
                    break

        # Mark missing names as absent (0)
        for name in names:
            if name not in attendance:
                attendance[name] = 0

        return attendance

    if st.button("📍 Detect Attendance", key="image_button"):
        if screenshot:
            with st.spinner("🧠 Extracting names with Gemini..."):
                image_bytes = screenshot.getvalue()
                all_names = extract_names_from_full_image(image_bytes)

            if all_names and not all_names[0].startswith("❌ Error"):
                with st.spinner("🧑‍💻 Detecting faces and matching attendance..."):
                    att = detect_attendance_with_global_names(image_cv, all_names, int(rows), int(cols))
                    df = pd.DataFrame(att.items(), columns=["Name", "Attendance Score"])
                    st.success("✅ Attendance Processed!")
                    st.dataframe(df)
                    st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="attendance.csv")
            else:
                st.error(f"Error extracting names: {all_names[0] if all_names else 'Unknown error'}")
        else:
            st.warning("Please upload a screenshot image first.")


import io

######################################
# 🎥 TAB 2 — Video Attendance (UPDATED)
######################################

with tab2:
    st.markdown("### 📥 Upload or Link a Meeting Recording")
    video_file = st.file_uploader("Upload Meeting Recording (.mp4)", type=["mp4"])
    gdrive_link = st.text_input("Or paste a public Google Drive video link:")

    interval = st.number_input("Frame Check Interval (minutes)", min_value=1, max_value=60, value=10)

    def frame_to_jpeg_bytes(frame):
        is_success, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes() if is_success else None

    def extract_names_from_frame_image(frame):
        img_bytes = frame_to_jpeg_bytes(frame)
        if not img_bytes:
            return []
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = """
        You are an AI helping to detect student names in a classroom screenshot (Google Meet or Zoom grid). 
        Extract all names **exactly as they appear** on screen, in order (left to right, top to bottom). 
        List only the names, one per line. No other text.
        """
        try:
            response = model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": img_bytes}
            ])
            return [line.strip() for line in response.text.strip().split("\n") if line.strip()]
        except Exception as e:
            st.error(f"Error extracting names: {str(e)}")
            return []

    def extract_frames(video_path, interval_seconds):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_seconds)

        frames, timestamps = [], []
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frame_small = cv2.resize(frame, (640, 360))
                timestamp = str(timedelta(seconds=int(count / fps)))
                frames.append(frame_small)
                timestamps.append(timestamp)
            count += 1

        cap.release()
        return frames, timestamps

    def detect_changes_over_time(name_snapshots, timestamps):
        logs = []
        all_names_set = set(name for snap in name_snapshots for name in snap)
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

    def download_gdrive_file(gdrive_url):
        import requests
        import re
        file_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', gdrive_url)
        if not file_id_match:
            st.error("Invalid Google Drive link.")
            return None

        file_id = file_id_match.group(1)
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                response = requests.get(download_url, stream=True)
                if response.status_code == 200:
                    for chunk in response.iter_content(1024 * 1024):
                        tmpfile.write(chunk)
                    return tmpfile.name
                else:
                    st.error("Failed to download from Google Drive.")
                    return None
        except Exception as e:
            st.error(f"Error downloading file: {e}")
            return None

    final_video_path = None

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(video_file.read())
            final_video_path = tmpfile.name

    elif gdrive_link:
        st.info("📥 Downloading video from Google Drive...")
        final_video_path = download_gdrive_file(gdrive_link)

    if final_video_path:
        st.success("🎬 Video ready. Processing...")

        with st.spinner("🔍 Extracting frames and detecting names..."):
            frames, timestamps = extract_frames(final_video_path, interval * 60)
            name_snapshots = []
            logs_table = []

            progress_bar = st.progress(0)

            for i, frame in enumerate(frames):
                names = extract_names_from_frame_image(frame)
                name_snapshots.append(names)
                logs_table.append({
                    "Timestamp": timestamps[i],
                    "Names Detected": ", ".join(names) if names else "No names detected"
                })
                progress_bar.progress((i + 1) / len(frames))

            event_logs = detect_changes_over_time(name_snapshots, timestamps)

        os.unlink(final_video_path)

        st.subheader("📌 Names at Each Interval")
        df1 = pd.DataFrame(logs_table)
        st.dataframe(df1)
        st.download_button("📥 Download Name Snapshots CSV", df1.to_csv(index=False), file_name="name_snapshots.csv")

        st.subheader("📈 Join/Leave Logs")
        df2 = pd.DataFrame(event_logs)
        st.dataframe(df2)
        st.download_button("📥 Download Join/Leave Logs CSV", df2.to_csv(index=False), file_name="join_leave_logs.csv")

        st.balloons()
        st.success("✅ Processing complete!")