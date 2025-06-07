import streamlit as st, requests, pandas as pd
from PIL import Image

st.title("📸 Attendance Detection via FastAPI")
file = st.file_uploader("Upload Screenshot", type=["jpg", "png"])
cols = st.number_input("Columns", min_value=1, value=5)
rows = st.number_input("Rows", min_value=1, value=3)

if file and st.button("📍 Detect Attendance"):
    st.image(file, caption="Uploaded")
    bytes_img = file.getvalue()
    with st.spinner("Calling FastAPI..."):
        try:
            res = requests.post("https://your-fastapi-url.com/process_attendance/",  # Change to your real URL
                                files={"file": ("screenshot.jpg", bytes_img, "image/jpeg")},
                                data={"rows": int(rows), "cols": int(cols)})
            result = res.json()
            if result["status"] == "success":
                df = pd.DataFrame(result["data"].items(), columns=["Name", "Attendance Score"])
                st.success("✅ Done!")
                st.dataframe(df)
                st.download_button("📥 Download CSV", df.to_csv(index=False), "attendance.csv")
            else:
                st.error(f"API Error: {result['message']}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
