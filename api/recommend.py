from http.client import HTTPException
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Dict, List

app = FastAPI(title="TernakPro AI Recommendation API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
class LivestockModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.animal_info = {}
        
    def load_model(self):
        """Load model dari file"""
        try:
            # Gunakan path absolut untuk Vercel
            model_path = os.path.join(os.path.dirname(__file__), '../models/rekomendasi_ternak_ternakpro_v1_model.pkl')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.encoders = model_data['encoders']
            self.feature_mapping = model_data['feature_mapping']
            
            # Default animal info jika tidak ada di model
            self.animal_info = {
                'ayam_pedaging': {
                    'name': 'Ayam Pedaging',
                    'initial_cost': 5000000,
                    'description': 'Ayam pedaging cocok untuk pemula dengan keuntungan cepat',
                    'feed_requirements': ['Konsentrat: 0.15kg/ekor/hari', 'Jagung: 0.1kg/ekor/hari'],
                    'health_risks': ['Penyakit ND', 'Penyakit AI'],
                    'tips': ['Vaksinasi teratur', 'Jaga kebersihan kandang']
                },
                'ayam_petelur': {
                    'name': 'Ayam Petelur',
                    'initial_cost': 6000000,
                    'description': 'Ayam petelur memberikan pendapatan rutin dari telur',
                    'feed_requirements': ['Konsentrat layer: 0.12kg/ekor/hari', 'Kalsium: 0.05kg/ekor/hari'],
                    'health_risks': ['Penyakit CRD', 'Tetelo'],
                    'tips': ['Pencahayaan cukup', 'Pemberian vitamin rutin']
                }
            }
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, input_data: Dict) -> Dict:
        """Membuat prediksi berdasarkan input"""
        try:
            # Preprocess input
            processed_input = self._preprocess_input(input_data)
            
            # Scale features
            scaled_input = self.scaler.transform([processed_input])
            
            # Predict
            prediction = self.model.predict(scaled_input)[0]
            success_rate, roi, market_demand = prediction
            
            # Get recommended animal
            recommended_animal = self._get_recommended_animal(input_data)
            
            # Get animal details
            animal_details = self.animal_info.get(recommended_animal, {})
            
            return {
                'recommended_animal': recommended_animal,
                'success_rate': float(success_rate),
                'roi': float(roi),
                'market_demand': float(market_demand),
                'animal_details': animal_details
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            raise
    
    def _preprocess_input(self, input_data: Dict) -> List:
        """Preprocess input data"""
        processed = []
        
        # Encode categorical features
        categorical_cols = ['region', 'goal', 'available_feed', 'time_availability', 'experience']
        
        for col in categorical_cols:
            if col in input_data:
                le = self.encoders.get(col)
                if le:
                    try:
                        # Coba encode value
                        if input_data[col] in le.classes_:
                            encoded = le.transform([input_data[col]])[0]
                        else:
                            # Fallback ke value pertama jika tidak ditemukan
                            encoded = 0
                    except Exception:
                        encoded = 0
                    processed.append(encoded)
                else:
                    processed.append(0)
            else:
                processed.append(0)
        
        # Add numerical features
        processed.append(input_data.get('land_size', 0))
        
        return processed
    
    def _get_recommended_animal(self, input_data: Dict) -> str:
        """Logic untuk rekomendasi ternak"""
        # Simple rule-based fallback jika model tidak memberikan rekomendasi spesifik
        goal = input_data.get('goal', '')
        land_size = input_data.get('land_size', 0)
        
        if goal == 'daging':
            if land_size < 100:
                return 'ayam_pedaging'
            elif land_size < 200:
                return 'kambing'
            else:
                return 'sapi_potong'
        elif goal == 'telur':
            return 'ayam_petelur'
        elif goal == 'susu':
            return 'sapi_perah'
        else:
            return 'ayam_pedaging'

# Initialize model
livestock_model = LivestockModel()

# Pydantic models untuk request validation
class RecommendationRequest(BaseModel):
    region: str
    land_size: float
    goal: str
    available_feed: str
    time_availability: str
    experience: str

# Load model ketika aplikasi start
@app.on_event("startup")
async def startup_event():
    try:
        livestock_model.load_model()
        print("AI Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.post("/api/recommend")
async def recommend_livestock(request: RecommendationRequest):
    """Endpoint untuk rekomendasi ternak"""
    try:
        input_data = {
            'region': request.region,
            'land_size': request.land_size,
            'goal': request.goal,
            'available_feed': request.available_feed,
            'time_availability': request.time_availability,
            'experience': request.experience
        }
        
        prediction = livestock_model.predict(input_data)
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": livestock_model.model is not None,
        "service": "TernakPro AI Recommendation API"
    }

@app.get("/")
async def root():
    return {"message": "TernakPro AI Recommendation API", "status": "online"}

# Handler untuk Vercel
def handler(request):
    return app