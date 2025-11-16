# fast_api_app.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
import sqlite3
from datetime import datetime
from typing import List, Optional

app = FastAPI(title="House Price API with History")

# --- بارگذاری مدل ---
model = joblib.load("house_price_model.pkl")

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
