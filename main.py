import streamlit as st
import requests
import pandas as pd

st.title("Zoom/Meet Attendance Tracker")

tab1, tab2 = st.tabs(["📸 Screenshot Attendance", "🎥 Video Attendance"])

with tab1:
    # your existing code to upload screenshot, rows, cols
    pass  # (keep your existing code here)

with tab2:
    video_file = st.file_uploader("Upload Meeting Recording (.mp4)", type=["mp4"])
    interval = st.number_input("Frame Check Interval (minutes)", min_value=1, max_value=60, value=10)

    if video_file and st.button("📍 Detect Video Attendance"):
        with st.spinner("Processing video attendance via API..."):
            try:
                res = requests.post(
                    "https://your-fastapi-url.com/process_video_attendance/",
                    files={"file": ("video.mp4", video_file.getvalue(), "video/mp4")},
                    data={"interval_minutes": interval}
                )
                result = res.json()
                if result["status"] == "success":
                    # Show snapshots
                    snapshots = result["name_snapshots"]
                    df_snapshots = pd.DataFrame([
                        {"Timestamp": s["timestamp"], "Names Detected": ", ".join(s["names"])} for s in snapshots
                    ])
                    st.subheader("Names at Each Interval")
                    st.dataframe(df_snapshots)
                    st.download_button("📥 Download Name Snapshots CSV", df_snapshots.to_csv(index=False), "name_snapshots.csv")

                    # Show join/leave logs
                    logs = result["event_logs"]
                    df_logs = pd.DataFrame(logs)
                    st.subheader("Join/Leave Logs")
                    st.dataframe(df_logs)
                    st.download_button("📥 Download Join/Leave Logs CSV", df_logs.to_csv(index=False), "join_leave_logs.csv")

                    st.success("✅ Video attendance processed!")
                else:
                    st.error(f"API Error: {result.get('message','Unknown error')}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
