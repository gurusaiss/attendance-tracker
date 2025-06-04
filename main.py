# app.py
import streamlit as st
import requests
import pandas as pd
from PIL import Image
import numpy as np

st.set_page_config(page_title="Zoom/Meet Attendance Tracker", layout="wide")
st.title("📸 Zoom/Meet Attendance Tracker")

screenshot = st.file_uploader("Upload Screenshot of Zoom/Meet", type=["png", "jpg", "jpeg"])
cols = st.number_input("Enter No. of Columns (People per Row)", min_value=1, max_value=15, value=5)
rows = st.number_input("Enter No. of Rows", min_value=1, max_value=10, value=3)

API_URL = "http://localhost:8000/process_attendance/"  # Replace with public URL if needed

if st.button("📍 Detect Attendance"):
    if screenshot:
        with st.spinner("Uploading image to backend..."):
            files = {"file": screenshot}
            data = {"rows": int(rows), "cols": int(cols)}

            try:
                res = requests.post(API_URL, files=files, data=data)
                if res.ok:
                    json_data = res.json()
                    if json_data["status"] == "success":
                        att_dict = json_data["data"]
                        df = pd.DataFrame(att_dict.items(), columns=["Name", "Attendance Score"])
                        st.success("✅ Attendance Processed!")
                        st.dataframe(df)
                        st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="attendance.csv")
                    else:
                        st.error("❌ API Error: " + json_data["message"])
                else:
                    st.error(f"❌ API Call Failed: {res.status_code}")
            except Exception as e:
                st.error("❌ Request Failed: " + str(e))
    else:
        st.warning("Please upload a screenshot image first.")
