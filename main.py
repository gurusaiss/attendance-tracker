import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd

st.title("📸 Attendance Detection via FastAPI")

# File uploader
uploaded_file = st.file_uploader("Upload Zoom/Meet Screenshot", type=["jpg", "jpeg", "png"])
cols = st.number_input("Columns", min_value=1, value=5)
rows = st.number_input("Rows", min_value=1, value=3)

if uploaded_file and st.button("📍 Detect Attendance"):
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image")

    # Convert to bytes for API call
    image_bytes = uploaded_file.getvalue()

    # Send to your FastAPI backend
    with st.spinner("Calling backend API..."):
        try:
            res = requests.post(
                "https://your-fastapi-url.com/process_attendance/",  # <- your deployed API
                files={"file": ("screenshot.jpg", image_bytes, "image/jpeg")},
                data={"rows": int(rows), "cols": int(cols)}
            )

            result = res.json()

            if result["status"] == "success":
                df = pd.DataFrame(result["data"].items(), columns=["Name", "Attendance Score"])
                st.success("✅ Attendance Detected!")
                st.dataframe(df)
                st.download_button("Download CSV", df.to_csv(index=False), "attendance.csv")
            else:
                st.error(f"API Error: {result['message']}")

        except Exception as e:
            st.error(f"❌ Request failed: {e}")
