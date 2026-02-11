"""
Updated AI-Powered Symptom Analyzer
Uses trained ML model from your disease-symptom dataset
"""
import os
import re
import pickle
import joblib
import numpy as np
from typing import Dict, List
from langdetect import detect, LangDetectException

class MLSymptomAnalyzer:
    """
    Analyzes symptoms using ML model trained on your dataset
    """
    
    def __init__(self, model_path='models'):
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.symptom_columns = None
        self.specialty_mapping = None
        self.symptom_keywords = {}
        
        # Load the trained model
        self._load_model()
        
        # Initialize symptom keyword mappings for text-to-vector conversion
        self._initialize_symptom_keywords()
    
    def _load_model(self):
        """Load the trained model and metadata"""
        try:
            # Load model
            model_file = os.path.join(self.model_path, 'disease_predictor.pkl')
            self.model = joblib.load(model_file)
            
            # Load label encoder
            encoder_file = os.path.join(self.model_path, 'label_encoder.pkl')
            self.label_encoder = joblib.load(encoder_file)
            
            # Load symptom columns
            symptoms_file = os.path.join(self.model_path, 'symptom_columns.pkl')
            with open(symptoms_file, 'rb') as f:
                self.symptom_columns = pickle.load(f)
            
            # Load specialty mapping
            mapping_file = os.path.join(self.model_path, 'specialty_mapping.pkl')
            with open(mapping_file, 'rb') as f:
                self.specialty_mapping = pickle.load(f)
            
            print(f"✓ Model loaded successfully with {len(self.symptom_columns)} symptoms")
            
        except FileNotFoundError as e:
            print(f"⚠ Model files not found. Please run train_disease_model.py first!")
            print(f"Error: {e}")
            # Fall back to keyword-based system
            self._use_fallback_system()
    
    def _use_fallback_system(self):
        """Use keyword-based system if model not trained"""
        print("Using fallback keyword-based system...")
        from symptom_analyzer import SymptomAnalyzer
        self.fallback = SymptomAnalyzer()
    
    def _initialize_symptom_keywords(self):
        """
        Create mapping from symptom names to keywords for matching user input
        This converts symptom column names to searchable keywords
        """
        if self.symptom_columns is None:
            return
        
        for symptom in self.symptom_columns:
            # Clean symptom name and create variations
            clean_name = symptom.lower().strip()
            
            # Create keyword variations
            keywords = [clean_name]
            
            # Add variations without "or"
            if ' or ' in clean_name:
                keywords.extend(clean_name.split(' or '))
            
            # Add single words
            words = clean_name.replace('(', '').replace(')', '').split()
            keywords.extend(words)
            
            self.symptom_keywords[symptom] = keywords
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        try:
            lang = detect(text)
            if lang in ['hi', 'te', 'en']:
                return lang
            return 'en'
        except LangDetectException:
            return 'en'
    
    def text_to_symptom_vector(self, symptoms_text: str) -> np.ndarray:
        """
        Convert user's text description to symptom vector
        Matches keywords in text to symptom columns
        """
        # Initialize vector with zeros
        symptom_vector = np.zeros(len(self.symptom_columns))
        
        # Preprocess text
        text_lower = symptoms_text.lower()
        
        # Match each symptom
        for idx, symptom in enumerate(self.symptom_columns):
            keywords = self.symptom_keywords.get(symptom, [symptom.lower()])
            
            # Check if any keyword is in the text
            for keyword in keywords:
                if keyword in text_lower:
                    symptom_vector[idx] = 1
                    break
        
        return symptom_vector.reshape(1, -1)
    
    def predict_disease(self, symptom_vector: np.ndarray) -> Dict:
        """
        Predict disease from symptom vector
        Returns disease name and probability
        """
        # Get prediction
        prediction = self.model.predict(symptom_vector)[0]
        probabilities = self.model.predict_proba(symptom_vector)[0]
        
        # Get disease name
        disease_name = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence
        confidence = probabilities[prediction]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities)[::-1][:3]
        top_3_diseases = []
        for idx in top_3_idx:
            disease = self.label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx]
            top_3_diseases.append({'disease': disease, 'probability': float(prob)})
        
        return {
            'predicted_disease': disease_name,
            'confidence': float(confidence),
            'top_predictions': top_3_diseases
        }
    
    def get_specialty_for_disease(self, disease_name: str) -> str:
        """Map disease to medical specialty"""
        disease_lower = disease_name.lower()
        
        # Check specialty mapping first
        if disease_lower in self.specialty_mapping:
            return self.specialty_mapping[disease_lower]
        
        # Psychiatric/Mental Health
        if any(word in disease_lower for word in ['panic', 'anxiety', 'depression', 'psychiatric', 
                                                    'mental', 'bipolar', 'schizophrenia', 'phobia']):
            return 'Psychiatry'
        
        # Cardiac
        elif any(word in disease_lower for word in ['heart', 'cardiac', 'cardio', 'arrhythmia']):
            return 'Cardiology'
        
        # Neurological
        elif any(word in disease_lower for word in ['brain', 'neuro', 'stroke', 'seizure', 
                                                      'epilepsy', 'alzheimer', 'parkinson']):
            return 'Neurology'
        
        # Orthopedic
        elif any(word in disease_lower for word in ['bone', 'joint', 'ortho', 'fracture', 
                                                      'arthritis', 'osteo']):
            return 'Orthopedics'
        
        # Gastro
        elif any(word in disease_lower for word in ['stomach', 'gastro', 'digest', 'ulcer', 
                                                      'liver', 'intestine', 'colon']):
            return 'Gastroenterology'
        
        # Respiratory
        elif any(word in disease_lower for word in ['lung', 'breath', 'pulmo', 'asthma', 
                                                      'pneumonia', 'bronch']):
            return 'Pulmonology'
        
        # Dermatology
        elif any(word in disease_lower for word in ['skin', 'derma', 'rash', 'eczema', 
                                                      'psoriasis', 'acne']):
            return 'Dermatology'
        
        # Endocrinology
        elif any(word in disease_lower for word in ['diabetes', 'thyroid', 'hormone', 'endocrine']):
            return 'Endocrinology'
        
        # Urology
        elif any(word in disease_lower for word in ['kidney', 'bladder', 'urin', 'prostate', 'renal']):
            return 'Urology'
        
        # Pediatrics
        elif any(word in disease_lower for word in ['child', 'infant', 'pediatric', 'baby']):
            return 'Pediatrics'
        
        # Gynecology
        elif any(word in disease_lower for word in ['women', 'pregnancy', 'gyneco', 'menstr', 
                                                      'uterus', 'ovary']):
            return 'Gynecology'
        
        # ENT
        elif any(word in disease_lower for word in ['ear', 'nose', 'throat', 'sinus', 'tonsil']):
            return 'ENT'
        
        # Ophthalmology
        elif any(word in disease_lower for word in ['eye', 'vision', 'cataract', 'glaucoma', 
                                                      'ophthalm']):
            return 'Ophthalmology'
        
        else:
            return 'General Medicine'
    
    def determine_priority(self, disease_name: str, confidence: float) -> str:
        """Determine urgency level"""
        disease_lower = disease_name.lower()
        
        # Critical - Life threatening conditions
        critical_keywords = ['heart attack', 'stroke', 'sepsis', 'trauma', 'bleeding', 
                           'unconscious', 'cardiac arrest', 'respiratory failure']
        if any(keyword in disease_lower for keyword in critical_keywords):
            return 'critical'
        
        # Urgent - Serious conditions requiring prompt attention
        urgent_keywords = ['heart disease', 'arrhythmia', 'seizure', 'chest pain', 
                         'severe', 'acute', 'infection']
        if any(keyword in disease_lower for keyword in urgent_keywords):
            return 'urgent'
        
        # Panic disorder and anxiety - urgent but not critical
        if 'panic' in disease_lower or 'anxiety' in disease_lower:
            return 'urgent'  # Mental health crises need prompt care
        
        # High confidence + concerning symptoms = more urgent
        if confidence > 0.85:
            return 'urgent'
        
        # Default to normal
        return 'normal'
    
    def analyze_symptoms(self, symptoms_text: str, language: str = None) -> Dict:
        """
        Main method to analyze symptoms
        
        Args:
            symptoms_text: User's symptom description
            language: Language code (en, hi, te) - auto-detected if None
            
        Returns:
            Dictionary with disease prediction, specialty, and priority
        """
        # Detect language if not provided
        if language is None:
            language = self.detect_language(symptoms_text)
        
        # Convert text to symptom vector
        symptom_vector = self.text_to_symptom_vector(symptoms_text)
        
        # Check if any symptoms were detected
        if symptom_vector.sum() == 0:
            return {
                'category': 'general_medicine',
                'specialty': 'General Medicine',
                'confidence': 0.5,
                'priority': 'normal',
                'language': language,
                'original_text': symptoms_text,
                'message': 'No specific symptoms detected. Please consult a general physician.'
            }
        
        # Predict disease
        prediction_result = self.predict_disease(symptom_vector)
        
        # Get specialty
        specialty = self.get_specialty_for_disease(prediction_result['predicted_disease'])
        
        # Determine priority
        priority = self.determine_priority(
            prediction_result['predicted_disease'],
            prediction_result['confidence']
        )
        
        return {
            'category': specialty.lower().replace(' ', '_'),
            'specialty': specialty,
            'predicted_disease': prediction_result['predicted_disease'],
            'confidence': prediction_result['confidence'],
            'top_predictions': prediction_result['top_predictions'],
            'priority': priority,
            'language': language,
            'original_text': symptoms_text,
            'symptoms_detected': int(symptom_vector.sum())
        }
    
    def get_related_specialties(self, category: str) -> List[str]:
        """Get related medical specialties for a category"""
        specialty_relations = {
            'cardiology': ['Cardiology', 'Internal Medicine', 'Emergency Medicine'],
            'neurology': ['Neurology', 'Neurosurgery', 'Emergency Medicine'],
            'orthopedics': ['Orthopedics', 'Sports Medicine', 'Physiotherapy'],
            'gastroenterology': ['Gastroenterology', 'General Surgery', 'Internal Medicine'],
            'pulmonology': ['Pulmonology', 'Internal Medicine', 'Emergency Medicine'],
            'dermatology': ['Dermatology', 'Allergy & Immunology'],
            'emergency': ['Emergency Medicine', 'Trauma Care', 'Critical Care'],
            'pediatrics': ['Pediatrics', 'Neonatology'],
            'gynecology': ['Gynecology', 'Obstetrics'],
            'general_medicine': ['General Medicine', 'Internal Medicine', 'Family Medicine']
        }
        
        return specialty_relations.get(category, ['General Medicine'])


# Singleton instance
ml_symptom_analyzer = MLSymptomAnalyzer()
symptom_analyzer = ml_symptom_analyzer