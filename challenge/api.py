import fastapi
from fastapi import FastAPI, status, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from challenge.model import DelayModel
import pandas as pd
from fastapi.concurrency import run_in_threadpool




from fastapi.responses import JSONResponse
from json import JSONDecodeError # Necesitamos capturar errores de JSON



VALID_AIRLINES = ['American Airlines',
	 'Air Canada',
	 'Air France',
	 'Aeromexico',
	 'Aerolineas Argentinas',
	 'Austral',
	 'Avianca',
	 'Alitalia',
	 'British Airways',
	 'Copa Air',
	 'Delta Air',
	 'Gol Trans',
	 'Iberia',
	 'K.L.M.',
	 'Qantas Airways',
	 'United Airlines',
	 'Grupo LATAM',
	 'Sky Airline',
	 'Latin American Wings',
	 'Plus Ultra Lineas Aereas',
	 'JetSmart SPA',
	 'Oceanair Linhas Aereas',
	 'Lacsa'
 ]

class AirlinesEnum(str, Enum):
    """Enum for airlines."""
    American_Airlines='American Airlines'
    Air_Canada='Air Canada'
    Air_France='Air France'
    Aeromexico='Aeromexico'
    Aerolineas_Argentinas='Aerolineas Argentinas'
    Austral='Austral'
    Avianca='Avianca'
    Alitalia='Alitalia'
    British_Airways='British Airways'
    Copa_Air='Copa Air'
    Delta_Air='Delta Air'
    Gol_Trans='Gol Trans'
    Iberia='Iberia'
    KLM='K.L.M.'
    Qantas_Airways='Qantas Airways'
    United_Airlines='United Airlines'
    Grupo_LATAM='Grupo LATAM'
    Sky_Airline='Sky Airline'
    Latin_American_Wings='Latin American Wings'
    Plus_Ultra='Plus Ultra Lineas Aereas'
    JetSmart_SPA='JetSmart SPA'
    Oceanair='Oceanair Linhas Aereas'
    Lacsa='Lacsa'


class FlightTypeEnum(str, Enum):
    """Enum for valid flight types."""
    NATIONAL = "N"
    INTERNATIONAL = "I"

class Flight(BaseModel):

    OPERA: AirlinesEnum = Field(..., alias="OPERA")
    TIPOVUELO: FlightTypeEnum = Field(..., alias="TIPOVUELO")
    MES: int = Field(..., ge=1, le=12, description="The month of the flight, must be between 1 and 12", alias="MES")
    
  
    class Config:

        use_enum_values = True

        allow_population_by_field_name = False

class FlightsPayload(BaseModel):
    flights: List[Flight]

class PredictionResponse(BaseModel):
    predict: List[int]


app = fastapi.FastAPI(title="Flight Delays Prediction API",
    version="1.0.0")
    
    
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()},
    )


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict2", status_code=200)
async def post_predict2(payload: FlightsPayload) -> PredictionResponse:
    model = DelayModel()
    
    if not payload.flights:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The flights list cannot be empty."
        )


    list_of_flights = [flight.dict() for flight in payload.flights]



    
    features = model.preprocess(
            data=pd.DataFrame(list_of_flights)
        )
        
    
    predictions = model.predict(features)

    return PredictionResponse(predict=predictions)
    


@app.post(
    "/predict",
    summary="Performs a prediction for flights delays",
)
async def post_predict(request: Request) -> dict:
  
    
    model = DelayModel()

    try:
        data = await request.json()
    except JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON format."
        )


    if not isinstance(data, dict) or "flights" not in data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request body must be a JSON object with a 'flights' key."
        )

    flights = data["flights"]
    if not isinstance(flights, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'flights' key must contain a list of flight objects."
        )

    if not flights:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The flights list cannot be empty."
        )


    for i, flight in enumerate(flights):

        if not isinstance(flight, dict):
             raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Item at index {i} in 'flights' is not a valid object."
            )

        required_keys = {"OPERA", "TIPOVUELO", "MES"}
        if not required_keys.issubset(flight.keys()):
            missing = required_keys - flight.keys()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing keys in flight at index {i}: {', '.join(missing)}"
            )
        

        if flight["OPERA"] not in VALID_AIRLINES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid 'OPERA' in flight at index {i}: {flight['OPERA']}"
            )
        if flight["TIPOVUELO"] not in ["N", "I"]:
             raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid 'TIPOVUELO' in flight at index {i}: {flight['TIPOVUELO']}"
            )
        if not (isinstance(flight["MES"], int) and 1 <= flight["MES"] <= 12):
             raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid 'MES' in flight at index {i}: {flight['MES']}. Must be an integer between 1 and 12."
            )


   

    features = model.preprocess(
            data=pd.DataFrame(flights)
        )
        
    #prediction = await run_in_threadpool(blocking_prediction_logic, flights_data)
 
    prediction = model.predict(features)
    #print("this is prediction")
    #print("...............................................")
    return {"predict": prediction} 
    #return prediction
	

