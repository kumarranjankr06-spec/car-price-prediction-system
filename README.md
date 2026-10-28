# 🚗 Car Price Prediction

Predicts the resale price of a used car (in ₹ lakhs) from its details — brand, year, kilometers driven, fuel type, mileage, engine, power and so on.

A scikit-learn pipeline (data cleaning → scaling/encoding → Random Forest) is served by a FastAPI backend, with a single-page brutalist-style HTML frontend.

## Project structure

```
.
├── main.py                       # FastAPI app: loads the model, exposes POST /prediction
├── mnop.py                       # `changes` transformer: cleans raw input (brand from Name, numbers from "18.5 kmpl" etc.)
├── car_price_pipeline.pkl        # Trained pipeline (created by the notebook)
├── Untitled1-checkpoint.ipynb    # Notebook: data cleaning, training, saving the model
├── index.html                    # Frontend (HTML + CSS + JS, no build step)
├── data/cars.csv                 # Training data
└── requirements.txt              # Backend dependencies
```

## How the model works

1. **`changes`** (`mnop.py`): takes the raw car details and
   - extracts `Brand` from `Name` (`"Hyundai i20 Sportz"` → `"Hyundai"`) and drops `Name`
   - turns `Mileage`, `Engine`, `Power` text into numbers (`"1197 CC"` → `1197`)
2. **ColumnTransformer**
   - numeric (`Year`, `Kilometers_Driven`, `Mileage`, `Engine`, `Power`, `Seats`): fill missing with the mean, then standard scaling
   - categorical (`Brand`, `Location`, `Fuel_Type`, `Transmission`, `Owner_Type`): fill missing with the most frequent value, then one-hot encoding
3. **RandomForestRegressor** (200 trees) predicts the price.

R² on the test split is about 0.9.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

The API runs at `http://127.0.0.1:8000`; interactive docs are at `http://127.0.0.1:8000/docs`.

Then open `index.html` in a browser (for example with the VS Code **Live Server** extension), fill the form and click **Predict Price**.

## API

### `POST /prediction`

Request body (JSON):

```json
{
  "Name": "Mahindra Scorpio",
  "Location": "Pune",
  "Year": 2012,
  "Kilometers_Driven": 99000,
  "Fuel_Type": "Diesel",
  "Transmission": "Manual",
  "Owner_Type": "Third",
  "Mileage": "12.05 kmpl",
  "Engine": "2179 CC",
  "Power": "120 bhp",
  "Seats": 8
}
```

`Mileage`, `Engine`, `Power` and `Seats` may be `null`; they are filled in by the pipeline.

Response:

```json
{ "predicted_price": 6.06 }
```

### `POST /home`

Health check — returns a short message saying the model is running.

## Retraining the model

Open `Untitled1-checkpoint.ipynb` and run all cells. It reads `data/cars.csv`, trains the pipeline and overwrites `car_price_pipeline.pkl`.

The notebook imports `changes` from `mnop.py` instead of defining it, so the saved pipeline can be loaded by `main.py`. If you change the cleaning logic, change it in `mnop.py` and restart the notebook kernel.

The notebook also needs `matplotlib`, `seaborn` and `jupyter`, which are not in `requirements.txt` since the server doesn't use them.

## Deploying on Render

### Backend (Web Service)

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Environment variable | `PYTHON_VERSION` = `3.13.9` |

Notes:

- `car_price_pipeline.pkl` must be in the repo, since the server loads it at startup. It is about 80 MB: GitHub accepts it (limit 100 MB) but warns about it. Lowering `n_estimators` in the notebook makes it smaller.
- `scikit-learn` is pinned to the version the model was trained with. If you upgrade it, retrain the model.
- On the free plan the service sleeps when idle, so the first request after a while can take up to a minute.

### Frontend (Static Site)

| Setting | Value |
|---|---|
| Build Command | *(leave empty)* |
| Publish Directory | `.` |

Before deploying, change `API_URL` in the `<script>` at the bottom of `index.html` from `http://127.0.0.1:8000/prediction` to your backend's Render URL, for example:

```js
const API_URL = "https://your-backend.onrender.com/prediction";
```

## Tech stack

Python · pandas · scikit-learn · FastAPI · Uvicorn · HTML/CSS/JavaScript
