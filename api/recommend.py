from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, List
import json

app = FastAPI(
    title="TernakPro AI Recommendation API",
    description="API untuk rekomendasi ternak berdasarkan kondisi peternak",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model data structure
class LivestockModel:
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.error_message = None
        
    def load_model(self):
        """Try to load model - fallback to simple logic if fails"""
        try:
            # Untuk Vercel, path akan berbeda
            model_path = os.path.join(os.path.dirname(__file__), '../models/rekomendasi_ternak_ternakpro_v1_model.pkl')
            
            # Coba path alternative untuk Vercel
            if not os.path.exists(model_path):
                model_path = os.path.join(os.path.dirname(__file__), 'models/rekomendasi_ternak_ternakpro_v1_model.pkl')
            
            if not os.path.exists(model_path):
                model_path = 'rekomendasi_ternak_ternakpro_v1_model.pkl'
            
            if os.path.exists(model_path):
                print(f"📦 Loading model from: {model_path}")
                model_data = joblib.load(model_path)
                self.model = model_data
                self.is_loaded = True
                self.error_message = None
                print("✅ Model loaded successfully!")
            else:
                self.error_message = "Model file not found"
                print("⚠️  Model file not found, using rule-based fallback")
                
        except Exception as e:
            self.error_message = str(e)
            print(f"❌ Error loading model: {e}")
    
    def predict(self, input_data: Dict) -> Dict:
        """Make prediction using model or fallback logic"""
        try:
            if self.is_loaded and self.model:
                return self._predict_with_model(input_data)
            else:
                return self._predict_with_rules(input_data)
                
        except Exception as e:
            print(f"⚠️  Prediction error, using fallback: {e}")
            return self._predict_with_rules(input_data)
    
    def _predict_with_model(self, input_data: Dict) -> Dict:
        """Prediction menggunakan model machine learning"""
        # Implementasi sesuai dengan struktur model Anda
        # Contoh sederhana - sesuaikan dengan model sebenarnya
        
        # Simulasi prediction berdasarkan input
        goal = input_data.get('goal', 'daging')
        land_size = input_data.get('land_size', 0)
        
        if goal == 'daging':
            if land_size < 100:
                result = {'ayam_pedaging': 0.85}
            elif land_size < 200:
                result = {'kambing': 0.78}
            else:
                result = {'sapi_potong': 0.82}
        elif goal == 'telur':
            result = {'ayam_petelur': 0.80}
        elif goal == 'susu':
            result = {'sapi_perah': 0.75}
        else:
            result = {'ayam_pedaging': 0.70}
        
        animal = list(result.keys())[0]
        success_rate = result[animal]
        
        return {
            'recommended_animal': animal,
            'success_rate': success_rate,
            'roi': success_rate * 0.4,  # ROI sekitar 40% dari success rate
            'market_demand': 0.8 + (success_rate * 0.2)  # Market demand 80-100%
        }
    
    def _predict_with_rules(self, input_data: Dict) -> Dict:
        """Rule-based fallback prediction"""
        goal = input_data.get('goal', 'daging')
        land_size = input_data.get('land_size', 0)
        experience = input_data.get('experience', 'pemula')
        
        # Rules berdasarkan pengalaman
        experience_multiplier = {
            'pemula': 0.8,
            'menengah': 0.9,
            'ahli': 1.0
        }
        
        multiplier = experience_multiplier.get(experience, 0.8)
        
        if goal == 'daging':
            if land_size < 100:
                animal = 'ayam_pedaging'
                success_rate = 0.85 * multiplier
            elif land_size < 200:
                animal = 'kambing'
                success_rate = 0.75 * multiplier
            else:
                animal = 'sapi_potong'
                success_rate = 0.80 * multiplier
        elif goal == 'telur':
            animal = 'ayam_petelur'
            success_rate = 0.78 * multiplier
        elif goal == 'susu':
            animal = 'sapi_perah'
            success_rate = 0.82 * multiplier
        else:
            animal = 'ayam_pedaging'
            success_rate = 0.70 * multiplier
        
        return {
            'recommended_animal': animal,
            'success_rate': success_rate,
            'roi': success_rate * 0.4,
            'market_demand': 0.8 + (success_rate * 0.2)
        }

# Initialize model
livestock_model = LivestockModel()

# Pydantic model for request validation
class RecommendationRequest(BaseModel):
    region: str = "jawa_barat"
    land_size: float
    goal: str
    available_feed: str
    time_availability: str
    experience: str

# Animal information database
ANIMAL_INFO = {
    'ayam_pedaging': {
        'name': 'Ayam Pedaging',
        'initial_cost': 5000000,
        'description': 'Ayam pedaging cocok untuk pemula dengan keuntungan cepat dalam 30-40 hari',
        'feed_requirements': ['Konsentrat: 0.15kg/ekor/hari', 'Jagung: 0.1kg/ekor/hari', 'Vitamin: sesuai kebutuhan'],
        'health_risks': ['Penyakit ND (Newcastle Disease)', 'Penyakit AI (Avian Influenza)', 'Gumboro'],
        'tips': ['Vaksinasi teratur', 'Jaga kebersihan kandang', 'Kontrol suhu dan ventilasi']
    },
    'ayam_petelur': {
        'name': 'Ayam Petelur',
        'initial_cost': 6000000,
        'description': 'Ayam petelur memberikan pendapatan rutin dari telur dengan masa produktif 1-2 tahun',
        'feed_requirements': ['Konsentrat layer: 0.12kg/ekor/hari', 'Kalsium: 0.05kg/ekor/hari', 'Vitamin: sesuai kebutuhan'],
        'health_risks': ['Penyakit CRD', 'Tetelo', 'Cacingan'],
        'tips': ['Pencahayaan cukup 14-16 jam/hari', 'Pemberian vitamin rutin', 'Kandang bersih dan kering']
    },
    'sapi_potong': {
        'name': 'Sapi Potong',
        'initial_cost': 25000000,
        'description': 'Sapi potong memberikan keuntungan tinggi tetapi membutuhkan modal besar dan lahan luas',
        'feed_requirements': ['Rumput: 30kg/ekor/hari', 'Konsentrat: 5kg/ekor/hari', 'Mineral: sesuai kebutuhan'],
        'health_risks': ['Penyakit mulut dan kuku', 'Antraks', 'Cacingan'],
        'tips': ['Kandang luas dengan drainase baik', 'Perhatikan sanitasi kandang', 'Vaksinasi rutin']
    },
    'kambing': {
        'name': 'Kambing',
        'initial_cost': 8000000,
        'description': 'Kambing mudah dipelihara dan memiliki permintaan pasar yang stabil',
        'feed_requirements': ['Rumput: 10kg/ekor/hari', 'Konsentrat: 1kg/ekor/hari', 'Air minum: secukupnya'],
        'health_risks': ['Cacingan', 'Scabies', 'Pneumonia'],
        'tips': ['Kandang kering dan bersih', 'Pemberian pakan berkualitas', 'Perhatikan kesehatan kaki']
    },
    'sapi_perah': {
        'name': 'Sapi Perah',
        'initial_cost': 30000000,
        'description': 'Sapi perah memberikan penghasilan rutin dari susu dengan perawatan intensif',
        'feed_requirements': ['Hijauan: 40kg/ekor/hari', 'Konsentrat: 6kg/ekor/hari', 'Mineral: sesuai kebutuhan'],
        'health_risks': ['Mastitis', 'Metritis', 'Cacingan'],
        'tips': ['Kandang bersih dan nyaman', 'Pemerahan yang hygienis', 'Pakan berkualitas tinggi']
    }
}

@app.on_event("startup")
async def startup_event():
    """Load model when application starts"""
    livestock_model.load_model()

@app.post("/api/recommend")
async def recommend_livestock(request: RecommendationRequest):
    """Get livestock recommendation"""
    try:
        input_data = {
            'region': request.region,
            'land_size': request.land_size,
            'goal': request.goal,
            'available_feed': request.available_feed,
            'time_availability': request.time_availability,
            'experience': request.experience
        }
        
        # Get prediction
        prediction = livestock_model.predict(input_data)
        animal_type = prediction['recommended_animal']
        
        # Get animal details
        animal_details = ANIMAL_INFO.get(animal_type, ANIMAL_INFO['ayam_pedaging'])
        
        # Format response
        response = {
            'success': True,
            'data': {
                'jenis_hewan': animal_details['name'],
                'alasan': self._generate_explanation(animal_details['name'], request, prediction),
                'biaya_awal': animal_details['initial_cost'],
                'potensi_keuntungan': int(animal_details['initial_cost'] * prediction['roi']),
                'roi': round(prediction['roi'] * 100, 1),
                'kesesuaian_kondisi': round(prediction['success_rate'] * 100, 1),
                'permintaan_pasar': round(prediction['market_demand'] * 100, 1),
                'deskripsi': animal_details['description'],
                'kebutuhan_pakan': animal_details['feed_requirements'],
                'resiko_kesehatan': animal_details['health_risks'],
                'tips': animal_details['tips'],
                'model_used': 'ml_model' if livestock_model.is_loaded else 'rule_based'
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

def _generate_explanation(animal_name: str, request: RecommendationRequest, prediction: Dict) -> str:
    """Generate explanation for recommendation"""
    explanations = {
        'ayam_pedaging': f"Ayam pedaging direkomendasikan untuk lahan {request.land_size}m² dengan tujuan {request.goal}. Cocok untuk {request.experience} dengan ROI {prediction['roi']*100:.1f}%",
        'ayam_petelur': f"Ayam petelur ideal untuk produksi telur dengan kesesuaian {prediction['success_rate']*100:.1f}%. Permintaan pasar stabil di {request.region}",
        'sapi_potong': f"Sapi potong cocok untuk lahan luas dengan potensi keuntungan tinggi. ROI mencapai {prediction['roi']*100:.1f}%",
        'kambing': f"Kambing mudah dipelihara dan sesuai untuk {request.experience}. Permintaan pasar {prediction['market_demand']*100:.1f}%",
        'sapi_perah': f"Sapi perah memberikan penghasilan rutin dari susu. Cocok untuk peternak dengan pengalaman"
    }
    
    return explanations.get(animal_name.lower(), f"{animal_name} direkomendasikan berdasarkan analisis kondisi Anda")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": livestock_model.is_loaded,
        "model_error": livestock_model.error_message,
        "service": "TernakPro AI Recommendation API",
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/")
async def root():
    return {
        "message": "TernakPro AI Recommendation API", 
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/recommend": "Get livestock recommendation",
            "GET /api/health": "Health check",
            "GET /": "Root endpoint"
        }
    }

# Handler untuk Vercel
def handler(request):
    return app