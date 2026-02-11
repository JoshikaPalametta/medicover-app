"""
Train ML Model from Disease-Symptom Dataset
This script trains a machine learning model using your CSV dataset
and replaces the keyword-based approach with a proper ML classifier
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

class DiseasePredictor:
    """
    Train and save a disease prediction model from your dataset
    """
    
    def __init__(self, dataset_path='Final_Augmented_dataset_Diseases_and_Symptoms.csv'):
        self.dataset_path = dataset_path
        self.model = None
        self.label_encoder = None
        self.symptom_columns = None
        self.disease_to_specialty = self._create_specialty_mapping()
        
    def _create_specialty_mapping(self):
        """
        Map diseases to medical specialties
        Customize this based on your needs
        """
        return {
            # Psychiatric/Mental Health
            'panic disorder': 'Psychiatry',
            'anxiety disorder': 'Psychiatry',
            'depression': 'Psychiatry',
            'bipolar disorder': 'Psychiatry',
            'schizophrenia': 'Psychiatry',
            
            # Cardiac/Cardiovascular
            'heart disease': 'Cardiology',
            'hypertension': 'Cardiology',
            'arrhythmia': 'Cardiology',
            'coronary artery disease': 'Cardiology',
            'heart failure': 'Cardiology',
            'myocardial infarction': 'Cardiology',
            
            # Neurological
            'stroke': 'Neurology',
            'migraine': 'Neurology',
            'seizure': 'Neurology',
            'epilepsy': 'Neurology',
            'alzheimer': 'Neurology',
            'parkinson': 'Neurology',
            'multiple sclerosis': 'Neurology',
            
            # Orthopedic
            'arthritis': 'Orthopedics',
            'fracture': 'Orthopedics',
            'osteoporosis': 'Orthopedics',
            'back pain': 'Orthopedics',
            'joint pain': 'Orthopedics',
            'osteoarthritis': 'Orthopedics',
            
            # Gastroenterology
            'gastritis': 'Gastroenterology',
            'ulcer': 'Gastroenterology',
            'ibs': 'Gastroenterology',
            'hepatitis': 'Gastroenterology',
            'crohn disease': 'Gastroenterology',
            'colitis': 'Gastroenterology',
            
            # Respiratory/Pulmonology
            'asthma': 'Pulmonology',
            'pneumonia': 'Pulmonology',
            'bronchitis': 'Pulmonology',
            'tuberculosis': 'Pulmonology',
            'copd': 'Pulmonology',
            'lung cancer': 'Pulmonology',
            
            # Endocrinology
            'diabetes': 'Endocrinology',
            'thyroid disease': 'Endocrinology',
            'hypothyroidism': 'Endocrinology',
            'hyperthyroidism': 'Endocrinology',
            
            # Urology
            'kidney stones': 'Urology',
            'uti': 'Urology',
            'urinary tract infection': 'Urology',
            'prostate': 'Urology',
            
            # Gynecology/Obstetrics
            'pregnancy complications': 'Gynecology',
            'pcos': 'Gynecology',
            'endometriosis': 'Gynecology',
            'menstrual disorder': 'Gynecology',
            
            # Dermatology
            'eczema': 'Dermatology',
            'psoriasis': 'Dermatology',
            'acne': 'Dermatology',
            'skin infection': 'Dermatology',
            
            # ENT
            'sinusitis': 'ENT',
            'tonsillitis': 'ENT',
            'ear infection': 'ENT',
            
            # Ophthalmology
            'cataract': 'Ophthalmology',
            'glaucoma': 'Ophthalmology',
            
            # Pediatrics
            'infant': 'Pediatrics',
            'child': 'Pediatrics',
            
            # Emergency
            'sepsis': 'Emergency Medicine',
            'trauma': 'Emergency Medicine',
            
            # Default
            'default': 'General Medicine'
        }
    
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(self.dataset_path)
        
        # First column is diseases, rest are symptoms
        self.symptom_columns = df.columns[1:].tolist()
        
        print(f"✓ Loaded dataset with {len(df)} samples")
        print(f"✓ Number of unique diseases: {df.iloc[:, 0].nunique()}")
        print(f"✓ Number of symptoms: {len(self.symptom_columns)}")
        
        return df
    
    def prepare_data(self, df):
        """Prepare features and labels"""
        print("\nPreparing data...")
        
        # Features (X): All symptom columns (0s and 1s)
        X = df.iloc[:, 1:].values
        
        # Labels (y): Disease names
        y = df.iloc[:, 0].values
        
        # Encode disease names to numeric labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Labels shape: {y_encoded.shape}")
        
        return X, y_encoded
    
    def train_model(self, X, y, test_size=0.2, min_samples_per_disease=5):
        """Train the disease prediction model"""
        print("\nChecking class distribution...")
        
        # Count samples per disease
        unique, counts = np.unique(y, return_counts=True)
        disease_counts = dict(zip(unique, counts))
        
        # Find diseases with too few samples
        rare_diseases = [disease for disease, count in disease_counts.items() 
                        if count < min_samples_per_disease]
        
        if rare_diseases:
            print(f"⚠ Found {len(rare_diseases)} diseases with < {min_samples_per_disease} samples")
            print(f"  These will be removed to ensure reliable training")
            
            # Filter out rare diseases
            mask = np.array([disease_counts[label] >= min_samples_per_disease for label in y])
            X = X[mask]
            y = y[mask]
            
            print(f"✓ Filtered dataset: {len(X)} samples, {len(np.unique(y))} diseases")
        
        print("\nSplitting data...")
        # Use stratify only if all diseases have enough samples
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError:
            # Fallback: split without stratification
            print("⚠ Cannot stratify (some diseases still too rare), using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        print("\nTraining Random Forest model...")
        print("⚠ Using reduced model complexity to avoid memory issues...")
        
        self.model = RandomForestClassifier(
            n_estimators=100,      # Reduced from 200 to save memory
            max_depth=15,          # Reduced from 20 to save memory
            min_samples_split=10,  # Increased to reduce tree size
            min_samples_leaf=5,    # Increased to reduce tree size
            max_features='sqrt',   # Use sqrt(n_features) instead of all
            random_state=42,
            n_jobs=4,              # Limit parallel jobs to save memory
            verbose=1              # Show progress
        )
        
        print("Training... (this may take a few minutes)")
        self.model.fit(X_train, y_train)
        
        # Evaluate in batches to avoid memory issues
        print("\nEvaluating model (in batches)...")
        
        batch_size = 5000
        all_predictions = []
        
        for i in range(0, len(X_test), batch_size):
            batch_end = min(i + batch_size, len(X_test))
            X_batch = X_test[i:batch_end]
            batch_pred = self.model.predict(X_batch)
            all_predictions.extend(batch_pred)
            print(f"  Processed {batch_end}/{len(X_test)} test samples")
        
        y_pred = np.array(all_predictions)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✓ Model Accuracy: {accuracy:.2%}")
        
        # Show class distribution in test set
        print(f"✓ Unique diseases in test set: {len(np.unique(y_test))}")
        
        # Show detailed metrics for a sample of diseases
        print("\n" + "="*60)
        print("Sample Classification Metrics (Top 20 Most Common Diseases):")
        print("="*60)
        
        # Get most common diseases in test set
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        top_diseases_idx = np.argsort(counts_test)[::-1][:20]  # Top 20 diseases
        top_diseases = unique_test[top_diseases_idx]
        
        # Get disease names
        disease_names = self.label_encoder.inverse_transform(top_diseases)
        
        # Filter predictions for these diseases
        mask = np.isin(y_test, top_diseases)
        
        if mask.sum() > 0:
            print(classification_report(
                y_test[mask], 
                y_pred[mask],
                labels=top_diseases,
                target_names=disease_names,
                zero_division=0
            ))
        
        return self.model
    
    def save_model(self, output_dir='models'):
        """Save the trained model and metadata"""
        print(f"\nSaving model to {output_dir}/...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(output_dir, 'disease_predictor.pkl')
        joblib.dump(self.model, model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Save label encoder
        encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, encoder_path)
        print(f"✓ Label encoder saved to {encoder_path}")
        
        # Save symptom columns
        symptoms_path = os.path.join(output_dir, 'symptom_columns.pkl')
        with open(symptoms_path, 'wb') as f:
            pickle.dump(self.symptom_columns, f)
        print(f"✓ Symptom columns saved to {symptoms_path}")
        
        # Save specialty mapping
        mapping_path = os.path.join(output_dir, 'specialty_mapping.pkl')
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.disease_to_specialty, f)
        print(f"✓ Specialty mapping saved to {mapping_path}")
        
        print("\n✅ Model training complete!")
    
    def get_feature_importance(self, top_n=20):
        """Get most important symptoms"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        
        print(f"\n{'='*60}")
        print(f"Top {top_n} Most Important Symptoms:")
        print(f"{'='*60}")
        for i, idx in enumerate(indices, 1):
            symptom = self.symptom_columns[idx]
            print(f"{i:2d}. {symptom[:50]:50s} - {importance[idx]:.4f}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("Disease Prediction Model Training")
    print("="*60)
    
    # Initialize trainer
    trainer = DiseasePredictor(
        dataset_path='Final_Augmented_dataset_Diseases_and_Symptoms.csv'
    )
    
    # Load data
    df = trainer.load_data()
    
    # Prepare data
    X, y = trainer.prepare_data(df)
    
    # Train model
    model = trainer.train_model(X, y)
    
    # Show feature importance
    trainer.get_feature_importance(top_n=30)
    
    # Save model
    trainer.save_model(output_dir='models')
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. The model is saved in the 'models/' directory")
    print("2. Update symptom_analyzer.py to use this model")
    print("3. Restart your Flask application")
    print("="*60)


if __name__ == '__main__':
    main()