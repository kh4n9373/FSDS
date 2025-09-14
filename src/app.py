# app.py
import asyncio, websockets, json, sqlite3, pickle
from fastapi import FastAPI, BackgroundTasks
from sklearn.linear_model import Ridge
import pandas as pd

DB = "crypto.db"
MODEL_FILE = "model.pkl"


# --- DB setup ---
conn = sqlite3.connect(DB, check_same_thread=False)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS ticks (ts INTEGER PRIMARY KEY, price REAL)")
cur.execute("CREATE TABLE IF NOT EXISTS preds (ts INTEGER PRIMARY KEY, price REAL, pred REAL)")
conn.commit()

# --- Collector task ---
async def collector():
    url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    async with websockets.connect(url, ping_interval=20) as ws:
        async for msg in ws:
            t = json.loads(msg)
            ts, price = int(t["T"]), float(t["p"])
            cur.execute("INSERT OR REPLACE INTO ticks VALUES (?,?)", (ts, price))
            conn.commit()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    asyncio.create_task(collector())
    asyncio.create_task(predictor_loop())
    yield

app = FastAPI(lifespan=lifespan)
# --- Trainer ---
@app.post("/train")
def train_model():
    df = pd.read_sql("SELECT * FROM ticks ORDER BY ts", conn)
    df["ret1"] = df["price"].pct_change()
    df = df.dropna()
    X = df[["ret1"]]
    y = df["price"].shift(-1).dropna()
    X, y = X.loc[y.index], y
    model = Ridge().fit(X, y)
    pickle.dump(model, open(MODEL_FILE, "wb"))
    return {"status": "trained", "n": len(X)}

# --- Predictor loop ---
import os

async def predictor_loop():
    while True:
        if not os.path.exists(MODEL_FILE):
            await asyncio.sleep(5)
            continue  # skip cho đến khi có model.pkl

        df = pd.read_sql("SELECT * FROM ticks ORDER BY ts DESC LIMIT 2", conn)
        if len(df) >= 2:
            model = pickle.load(open(MODEL_FILE, "rb"))
            ret1 = (df["price"].iloc[0] / df["price"].iloc[1]) - 1
            X = pd.DataFrame([[ret1]], columns=["ret1"])
            pred = float(model.predict(X)[0])
            ts, price = int(df["ts"].iloc[0]), float(df["price"].iloc[0])
            cur.execute("INSERT OR REPLACE INTO preds VALUES (?,?,?)", (ts, price, pred))
            conn.commit()
        await asyncio.sleep(1)


# --- Endpoints ---
@app.get("/ticks")
def get_ticks(limit: int = 100):
    rows = cur.execute("SELECT * FROM ticks ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return [{"ts": ts, "price": price} for ts, price in rows]

@app.get("/preds")
def get_preds(limit: int = 100):
    rows = cur.execute("SELECT * FROM preds ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return [{"ts": ts, "price": price, "pred": pred} for ts, price, pred in rows]

@app.get("/latest")
def get_latest():
    row = cur.execute("SELECT * FROM preds ORDER BY ts DESC LIMIT 1").fetchone()
    if not row: return {}
    ts, price, pred = row
    return {"ts": ts, "price": price, "pred": pred}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)