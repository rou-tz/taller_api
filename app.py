"""
Housing Rental Price Prediction API
FastAPI - Production Ready para PythonAnywhere
"""

import os
import logging
from typing import List, Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import pandas as pd

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOGGING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CARGA DEL MODELO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_housing_final.pkl')

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    FEATURES = model_data['features']  # ['surface', 'bedrooms', 'restrooms', 'ascensor', ...]
    Q99_THRESHOLD = model_data.get('Q99_threshold', float('inf'))
    FEATURE_IMPORTANCES = model_data.get('feature_importances', [])
    
    logger.info(f"✓ Modelo cargado correctamente")
    logger.info(f"  Features: {FEATURES}")
    logger.info(f"  Modelo: XGBRegressor con {len(FEATURES)} features")
except FileNotFoundError:
    logger.error(f"❌ Modelo no encontrado en {MODEL_PATH}")
    raise RuntimeError(f"Modelo no encontrado en {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Error al cargar modelo: {e}")
    raise RuntimeError(f"Error cargando modelo: {e}")

# Features booleanas — se convierten a 0/1
BOOL_FEATURES = [
    'ascensor', 'terraza', 'calefaccion',
    'aire_acondicionado', 'balcon', 'parking', 'piscina'
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# APP SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = FastAPI(
    title="Housing Rental Price API",
    description="Predice precios de alquiler de viviendas con XGBoost",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SCHEMAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PredictionRequest(BaseModel):
    """
    Request para predicción de precio de alquiler.
    
    Nota: location_encoded es un número (0-N) que representa la localización.
    Obtén los valores válidos en /location-info
    """
    surface: float = Field(..., gt=10, le=500, description="Superficie en m²", example=80.0)
    bedrooms: int = Field(..., ge=1, le=20, description="Número de habitaciones", example=3)
    restrooms: int = Field(..., ge=1, le=10, description="Número de baños", example=1)
    location_encoded: float = Field(..., description="Código de localización (ver /location-info)", example=1.0)
    ascensor: bool = Field(False, description="¿Tiene ascensor?")
    terraza: bool = Field(False, description="¿Tiene terraza?")
    calefaccion: bool = Field(False, description="¿Tiene calefacción?")
    aire_acondicionado: bool = Field(False, description="¿Tiene aire acondicionado?")
    balcon: bool = Field(False, description="¿Tiene balcón?")
    parking: bool = Field(False, description="¿Tiene parking?")
    piscina: bool = Field(False, description="¿Tiene piscina?")

    class Config:
        json_schema_extra = {
            "example": {
                "surface": 80.0,
                "bedrooms": 3,
                "restrooms": 1,
                "location_encoded": 1.0,
                "ascensor": True,
                "terraza": False,
                "calefaccion": True,
                "aire_acondicionado": False,
                "balcon": False,
                "parking": True,
                "piscina": False
            }
        }


class PredictionResponse(BaseModel):
    """Respuesta de predicción"""
    predicted_price: float
    currency: str = "EUR"
    note: str = "Predicción basada en modelo XGBoost"


class LocationInfoResponse(BaseModel):
    """Información sobre las localizaciones"""
    note: str
    features_required: List[str]


class HealthResponse(BaseModel):
    """Health check"""
    status: str
    model_loaded: bool
    features: int


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/")
def home():
    """Endpoint raíz"""
    return {
        'message': 'API de predicción de alquiler activa',
        'docs': 'Accede a /docs para documentación',
        'endpoints': {
            'GET /health': 'Health check',
            'GET /location-info': 'Info sobre location_encoded',
            'POST /predict': 'Predecir precio'
        }
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check - verifica que el servicio está funcionando"""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        features=len(FEATURES)
    )


@app.get("/location-info", response_model=LocationInfoResponse)
def location_info():
    """
    Info sobre cómo usar location_encoded.
    
    location_encoded es un número que codifica la localización.
    Debes conocer el mapeo de tu dataset original.
    
    Ejemplos de valores típicos: 0, 1, 2, 3, ... N
    """
    return LocationInfoResponse(
        note="location_encoded es un número (0-N) que representa la localización. Obtén el valor del mapeo de tu dataset.",
        features_required=FEATURES
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    """
    Predice el precio de alquiler de una vivienda.

    Body esperado (JSON):
    {
        "surface": 80.0,
        "bedrooms": 3,
        "restrooms": 1,
        "location_encoded": 1.0,
        "ascensor": true,
        "terraza": false,
        "calefaccion": true,
        "aire_acondicionado": false,
        "balcon": false,
        "parking": true,
        "piscina": false
    }

    Respuesta:
    {
        "predicted_price": 950.75,
        "currency": "EUR",
        "note": "Predicción basada en modelo XGBoost"
    }
    """
    try:
        # Construir diccionario de features EN EL ORDEN EXACTO DEL MODELO
        row = {
            'surface': float(data.surface),
            'bedrooms': int(data.bedrooms),
            'restrooms': int(data.restrooms),
            'ascensor': int(data.ascensor),
            'terraza': int(data.terraza),
            'calefaccion': int(data.calefaccion),
            'aire_acondicionado': int(data.aire_acondicionado),
            'balcon': int(data.balcon),
            'parking': int(data.parking),
            'piscina': int(data.piscina),
            'location_encoded': float(data.location_encoded),
        }

        # Crear DataFrame en el orden exacto que espera el modelo
        df_input = pd.DataFrame([row])[FEATURES]

        # Predicción
        prediction = model.predict(df_input)[0]
        price = round(float(prediction), 2)

        # Validar si está fuera del rango de entrenamiento
        if price > Q99_THRESHOLD:
            logger.warning(f"⚠️ Predicción por encima del Q99: {price}€")

        logger.info(f"✓ Predicción: {price}€ | location: {data.location_encoded}")

        return PredictionResponse(
            predicted_price=price,
            currency="EUR",
            note="Predicción basada en modelo XGBoost"
        )

    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error en datos: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error interno: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Error al procesar predicción: {str(e)}'
        )


@app.get("/model-info")
def model_info():
    """Información del modelo entrenado"""
    return {
        "model_type": "XGBRegressor",
        "features": FEATURES,
        "n_features": len(FEATURES),
        "bool_features": BOOL_FEATURES,
        "q99_threshold": float(Q99_THRESHOLD),
        "feature_importances": FEATURE_IMPORTANCES[:5]  # Top 5
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Para desarrollo local:
# uvicorn app:app --reload --host 0.0.0.0 --port 8000
#
# Para producción en PythonAnywhere:
# El WSGI file debe apuntar a este archivo y usar: from app import app as application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
