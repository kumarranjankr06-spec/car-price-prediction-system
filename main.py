from pathlib import Path 
from fastapi import FastAPI,HTTPException
import joblib
from mnop import changes  # needed so joblib can unpickle the pipeline
import pandas as pd 
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI(title="ml prediction system")

app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"],)

MODEL_PATH = Path(__file__).parent / "car_price_pipeline.pkl"
model = joblib.load(MODEL_PATH)

class CarFeatures(BaseModel):
    Name: str
    Location: str
    Year: int
    Kilometers_Driven: float
    Fuel_Type: str
    Transmission: str
    Owner_Type: str
    Mileage: str | float | None
    Engine: str | float | None
    Power: str | float | None
    Seats: float | None

  

@app.post("/home")
def home_page():
  return "the car price prediction model is running over here"

@app.post("/prediction")
def pred_page(car: CarFeatures):
  try:
    input_data=pd.DataFrame([car.model_dump()])
    
    prediction = model.predict(input_data)[0]

    return {
      "predicted_price": float(prediction)
    }
  except Exception:
    # Avoid exposing internal errors to the client.
    raise HTTPException(
      status_code=500,
      detail="Prediction failed. Check the backend logs."
    )
