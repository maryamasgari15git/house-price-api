# fast_api_app.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
import sqlite3
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import joblib
import pandas as pd
import io
from datetime import datetime
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
import openai
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



app = FastAPI(title="House Price API with History")
# --- Templates & Static Files ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# --- بارگذاری مدل ---
model = joblib.load("house_price_model.pkl")
REQUIRED_COLUMNS = ["area", "rooms", "distance"]


# --- مدل ورودی ---
class HouseData(BaseModel):
    area: float
    rooms: int
    distance: float

# --- مدل خروجی برای رکورد دیتابیس ---
class HistoryRecord(BaseModel):
    id: int
    timestamp: str
    area: float
    rooms: int
    distance: float
    predicted_price: float

# ---------- helper: init DB ----------
def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            area REAL,
            rooms INTEGER,
            distance REAL,
            predicted_price REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------- helper: read DB rows as dicts ----------
def row_to_dict(row):
    return {
        "id": row[0],
        "timestamp": row[1],
        "area": row[2],
        "rooms": row[3],
        "distance": row[4],
        "predicted_price": row[5]
    }
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------- Predict endpoint (ذخیره در DB) ----------
@app.post("/predict")
def predict_price(data: HouseData):
    X = [[data.area, data.rooms, data.distance]]
    prediction = float(model.predict(X)[0])

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (timestamp, area, rooms, distance, predicted_price)
        VALUES (?, ?, ?, ?, ?)
    """, (ts, data.area, data.rooms, data.distance, prediction))
    conn.commit()
    conn.close()

    return {"predicted_price": round(prediction, 2)}

# ---------- History endpoint: return all (with pagination & optional filters) ----------
@app.get("/history", response_model=List[HistoryRecord])
def get_history(
    limit: int = Query(100, gt=0, le=1000),
    offset: int = Query(0, ge=0),
    date_from: Optional[str] = Query(None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="YYYY-MM-DD")
):
    """
    Get history records.
    - limit, offset: pagination
    - date_from, date_to: optional date filters (inclusive), format YYYY-MM-DD
    """
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    query = "SELECT id, timestamp, area, rooms, distance, predicted_price FROM predictions"
    params = []
    where_clauses = []

    if date_from:
        where_clauses.append("date(timestamp) >= date(?)")
        params.append(date_from)
    if date_to:
        where_clauses.append("date(timestamp) <= date(?)")
        params.append(date_to)

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [row_to_dict(r) for r in rows]

# ---------- Get single record by id ----------
@app.get("/history/{record_id}", response_model=HistoryRecord)
def get_record(record_id: int):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, area, rooms, distance, predicted_price FROM predictions WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Record not found")
    return row_to_dict(row)

# ---------- Clear history (delete all) ----------
@app.delete("/history")
def clear_history(confirm: bool = Query(False, description="Must be true to actually clear history")):
    if not confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to clear history")
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    deleted = conn.total_changes
    conn.commit()
    conn.close()
    return {"deleted_rows": deleted}

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Please upload a CSV file (extension .csv).")

    contents = await file.read()
    try:
        s = contents.decode("utf-8")
    except UnicodeDecodeError:
        try:
            s = contents.decode("cp1252")
        except Exception:
            raise HTTPException(status_code=400, detail="Unable to decode the uploaded file. Use UTF-8 encoded CSV.")
    try:
        df = pd.read_csv(io.StringIO(s))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns in CSV: {missing}")

    try:
        df_required = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="raise")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV columns must be numeric. Error: {e}")

    X = df_required.values
    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    df_out = df.copy()
    df_out["predicted_price"] = preds.astype(float)


      # 🧩 ذخیره در دیتابیس
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for i, row in df_out.iterrows():
        cursor.execute("""
            INSERT INTO predictions (timestamp, area, rooms, distance, predicted_price)
            VALUES (?, ?, ?, ?, ?)
        """, (ts, row["area"], row["rooms"], row["distance"], row["predicted_price"]))
    
    conn.commit()
    conn.close()


    buf = io.BytesIO()
    df_out.to_csv(buf, index=False, encoding="utf-8")
    buf.seek(0)

    out_filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return StreamingResponse(buf, media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{out_filename}"'})
    

@app.post("/predict_with_explanation")
def explain_prediction(data: HouseData):
    # 1. ML Prediction
    X = [[data.area, data.rooms, data.distance]]
    prediction = float(model.predict(X)[0])

    # 2. LLM Explanation
    prompt = f"""
    You are a real estate assistant.
    The model predicted this price: {prediction}.
    Features:
    - Area: {data.area}
    - Rooms: {data.rooms}
    - Distance: {data.distance}

    Explain why the prediction makes sense in simple terms.
    """

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    explanation = response.output_text

    return {
        "predicted_price": round(prediction, 2),
        "explanation": explanation
    }