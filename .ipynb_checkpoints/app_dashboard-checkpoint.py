# app_dashboard.py
import streamlit as st
import pandas as pd
import requests
from io import StringIO
from datetime import datetime
import altair as alt

# ---------- تنظیمات ----------
st.set_page_config(page_title="🏡 House Price Dashboard", layout="wide")
st.title("🏡 House Price Dashboard")

API_URL = "http://127.0.0.1:8000"  # اگر سرورت روی آدرس/پورت دیگریه، اینو عوض کن

# ---------- ستون سمت راست - فرم ساده برای پیش‌بینی تکی ----------
with st.sidebar:
    st.header("🔮 Quick Predict (Single)")
    area = st.number_input("Area (m²)", min_value=1.0, max_value=10000.0, value=100.0, step=0.5)
    rooms = st.number_input("Rooms", min_value=1, max_value=50, value=3, step=1)
    distance = st.number_input("Distance (km)", min_value=0.0, max_value=500.0, value=5.0, step=0.1)
    if st.button("Predict (Single)"):
        payload = {"area": float(area), "rooms": int(rooms), "distance": float(distance)}
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if r.status_code == 200:
                res = r.json()
                st.success(f"🏠 Predicted Price: {res['predicted_price']:.2f}")
            else:
                st.error(f"API Error {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

st.markdown("---")

# ---------- ناحیهٔ اصلی: Tabs ----------
tab1, tab2, tab3 = st.tabs(["📂 CSV Batch", "📊 Dashboard / History", "⚙️ Utilities"])

# ---------- TAB 1: CSV Upload & Batch Predict ----------
with tab1:
    st.header("📂 Predict from CSV (batch)")
    st.markdown("Upload a CSV with columns: `area, rooms, distance`")
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = None

        if df is not None:
            st.subheader("Preview")
            st.dataframe(df.head())

            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("Run Batch Predict"):
                    try:
                        # prepare file payload for requests
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                        r = requests.post(f"{API_URL}/predict_csv", files=files, timeout=30)
                        if r.status_code == 200:
                            csv_text = r.content.decode("utf-8")
                            df_pred = pd.read_csv(StringIO(csv_text))
                            st.success("✅ Batch prediction completed")
                            st.dataframe(df_pred)
                            st.download_button("📥 Download predictions CSV", data=csv_text, file_name="predictions.csv", mime="text/csv")
                        else:
                            st.error(f"API Error {r.status_code}: {r.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
            with col2:
                st.info("Tips:\n- Ensure columns are named: area, rooms, distance\n- Use UTF-8 encoding if possible")

# ---------- TAB 2: Dashboard / History with filters and charts ----------
# ---------------- TAB 2: VIEW HISTORY ------------------
with tab2:
    st.header("📊 Prediction History")

    # دریافت تاریخچه از FastAPI
    try:
        r = requests.get(f"{API_URL}/history", timeout=10)
        r.raise_for_status()
        history_json = r.json()

        if len(history_json) == 0:
            st.info("No history found yet.")
        else:
            df_hist = pd.DataFrame(history_json)
            st.success(f"Loaded {len(df_hist)} previous predictions.")
            st.dataframe(df_hist)

            # ============================
            # 1) Scatter Plot with Altair
            # ============================
            st.subheader("🔵 Scatter: Area vs Predicted Price")

            scatter_chart = (
                alt.Chart(df_hist)
                .mark_circle(size=60)
                .encode(
                    x=alt.X("area", title="Area (m²)"),
                    y=alt.Y("predicted_price", title="Predicted Price"),
                    tooltip=["area", "rooms", "predicted_price", "distance", "timestamp"]
                )
                .interactive()
            )

            st.altair_chart(scatter_chart, use_container_width=True)

            # ============================
            # 2) Histogram
            # ============================
            st.subheader("📈 Predicted Price Distribution")

            hist_chart = (
                alt.Chart(df_hist)
                .mark_bar()
                .encode(
                    x=alt.X("predicted_price", bin=True),
                    y="count()",
                    tooltip=["count()"]
                )
            )

            st.altair_chart(hist_chart, use_container_width=True)

            # ============================
            # 3) Summary Statistics
            # ============================
            st.subheader("📌 Summary Statistics")
            st.write(df_hist[["area", "rooms", "distance", "predicted_price"]].describe())

            # ============================
            # 4) Clear history button
            # ============================
            st.subheader("🧹 Clear History")
            if st.button("Delete All History"):
                try:
                    rr = requests.delete(f"{API_URL}/history", params={"confirm": "true"}, timeout=10)
                    if rr.status_code == 200:
                        st.success("History deleted successfully.")
                    else:
                        st.error("Failed to delete history.")
                except Exception as e:
                    st.error(f"Error deleting history: {e}")

    except Exception as e:
        st.error(f"Connection error: {e}")

# ---------- TAB 3: Utilities ----------
with tab3:
    st.header("⚙️ Utilities")
    st.markdown("Small tools and admin actions.")

    if st.button("Refresh /health"):
        try:
            r = requests.get(f"{API_URL}/")
            if r.status_code == 200:
                st.success("API reachable.")
            else:
                st.error(f"API error: {r.status_code}")
        except Exception as e:
            st.error(f"Connection error: {e}")

    st.markdown("### Clear server history (dangerous)")
    st.warning("This will delete all records from server DB. Use only for testing.")
    confirm = st.checkbox("I understand and want to clear DB")
    if confirm and st.button("Delete all history"):
        try:
            r = requests.delete(f"{API_URL}/history", params={"confirm": "true"}, timeout=10)
            if r.status_code == 200:
                st.success(f"Deleted rows: {r.json().get('deleted_rows')}")
            else:
                st.error(f"Failed: {r.status_code} {r.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

# ---------- footer ----------
st.markdown("---")
st.caption("Built with FastAPI (backend) + Streamlit (dashboard).")
