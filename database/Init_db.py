"""
Database Initialization Script
Populates the database with sample hospital data for Visakhapatnam and surrounding areas
"""
import sys
import os
from datetime import datetime

# Add the backened directory to the Python path
# This allows us to import app and models
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
backend_dir = os.path.join(parent_dir, 'backened')  # Note: your folder is spelled 'backened'
sys.path.insert(0, backend_dir)

from app import app, db
from models import Hospital, Specialty, SymptomCategory


def init_specialties():
    """Initialize medical specialties"""
    specialties_data = [
        {
            'name': 'Cardiology',
            'name_te': 'హృదయ వైద్యం',
            'name_hi': 'हृदय रोग विज्ञान',
            'description': 'Heart and cardiovascular diseases',
            'keywords': ['heart', 'cardiac', 'cardiovascular', 'chest pain']
        },
        {
            'name': 'Neurology',
            'name_te': 'నాడీ వ్యాధుల విజ్ఞానం',
            'name_hi': 'तंत्रिका विज्ञान',
            'description': 'Brain and nervous system disorders',
            'keywords': ['brain', 'neurological', 'stroke', 'headache', 'seizure']
        },
        {
            'name': 'Orthopedics',
            'name_te': 'ఎముక వైద్యం',
            'name_hi': 'आर्थोपेडिक्स',
            'description': 'Bone, joint, and muscle disorders',
            'keywords': ['bone', 'fracture', 'joint pain', 'arthritis']
        },
        {
            'name': 'Gastroenterology',
            'name_te': 'జీర్ణాశయ వైద్యం',
            'name_hi': 'गैस्ट्रोएंटरोलॉजी',
            'description': 'Digestive system disorders',
            'keywords': ['stomach', 'digestion', 'abdominal pain', 'liver']
        },
        {
            'name': 'Pulmonology',
            'name_te': 'ఊపిరితిత్తుల వైద్యం',
            'name_hi': 'फुफ्फुसीय चिकित्सा',
            'description': 'Respiratory system disorders',
            'keywords': ['lung', 'breathing', 'asthma', 'cough', 'pneumonia']
        },
        {
            'name': 'Emergency Medicine',
            'name_te': 'అత్యవసర వైద్యం',
            'name_hi': 'आपातकालीन चिकित्सा',
            'description': 'Emergency and critical care',
            'keywords': ['emergency', 'trauma', 'accident', 'critical']
        },
        {
            'name': 'Pediatrics',
            'name_te': 'శిశు వైద్యం',
            'name_hi': 'बाल चिकित्सा',
            'description': 'Child healthcare',
            'keywords': ['child', 'infant', 'pediatric', 'baby']
        },
        {
            'name': 'Gynecology',
            'name_te': 'స్త్రీ రోగాల వైద్యం',
            'name_hi': 'स्त्री रोग विज्ञान',
            'description': "Women's health and obstetrics",
            'keywords': ['women', 'pregnancy', 'gynecology', 'obstetrics']
        },
    ]
    
    for spec_data in specialties_data:
        specialty = Specialty(**spec_data)
        db.session.add(specialty)
    
    db.session.commit()
    print("✓ Specialties initialized")


def init_symptom_categories():
    """Initialize symptom categories"""
    categories_data = [
        {
            'category': 'cardiac',
            'category_te': 'హృదయ సంబంధమైన',
            'category_hi': 'हृदय संबंधी',
            'description': 'Heart-related symptoms',
            'keywords': ['chest pain', 'heart attack', 'palpitations'],
            'related_specialties': ['Cardiology', 'Emergency Medicine'],
            'priority_level': 'urgent'
        },
        {
            'category': 'neurological',
            'category_te': 'నాడీ సంబంధమైన',
            'category_hi': 'तंत्रिका संबंधी',
            'description': 'Nervous system symptoms',
            'keywords': ['headache', 'stroke', 'seizure', 'numbness'],
            'related_specialties': ['Neurology', 'Emergency Medicine'],
            'priority_level': 'urgent'
        },
        {
            'category': 'orthopedic',
            'category_te': 'ఎముక సంబంధమైన',
            'category_hi': 'हड्डी संबंधी',
            'description': 'Bone and muscle symptoms',
            'keywords': ['bone pain', 'fracture', 'joint pain'],
            'related_specialties': ['Orthopedics'],
            'priority_level': 'normal'
        },
        {
            'category': 'emergency',
            'category_te': 'అత్యవసర',
            'category_hi': 'आपातकाल',
            'description': 'Emergency situations',
            'keywords': ['accident', 'severe pain', 'unconscious', 'bleeding'],
            'related_specialties': ['Emergency Medicine'],
            'priority_level': 'critical'
        },
    ]
    
    for cat_data in categories_data:
        category = SymptomCategory(**cat_data)
        db.session.add(category)
    
    db.session.commit()
    print("✓ Symptom categories initialized")


def init_hospitals():
    """Initialize sample hospitals in Visakhapatnam and nearby areas"""
    hospitals_data = [
        {
            'name': 'King George Hospital',
            'name_te': 'కింగ్ జార్జ్ ఆస్పత్రి',
            'name_hi': 'किंग जॉर्ज अस्पताल',
            'latitude': 17.7231,
            'longitude': 83.3015,
            'address': 'Maharani Peta, Visakhapatnam, Andhra Pradesh 530002',
            'address_te': 'మహారాణి పేట, విశాఖపట్నం, ఆంధ్ర ప్రదేశ్ 530002',
            'address_hi': 'महारानी पेटा, विशाखापत्तनम, आंध्र प्रदेश 530002',
            'city': 'Visakhapatnam',
            'state': 'Andhra Pradesh',
            'pincode': '530002',
            'phone': '+91-891-2565171',
            'emergency_phone': '108',
            'specialties': ['Emergency Medicine', 'General Surgery', 'Orthopedics', 'Cardiology', 
                          'Neurology', 'Pediatrics', 'Gynecology'],
            'services': ['24x7 Emergency', 'ICU', 'Operation Theater', 'Trauma Center', 'Blood Bank'],
            'facilities': ['ICU', 'Emergency Ward', 'Operation Theater', 'Blood Bank', 'Ambulance'],
            'is_24x7': True,
            'has_emergency': True,
            'has_ambulance': True,
            'bed_capacity': 800,
            'rating': 4.2,
            'total_reviews': 1250,
            'verified': True
        },
        {
            'name': 'GITAM Institute of Medical Sciences',
            'name_te': 'గీతం వైద్య శాస్త్ర సంస్థ',
            'name_hi': 'गीतम चिकित्सा विज्ञान संस्थान',
            'latitude': 17.7833,
            'longitude': 83.3786,
            'address': 'Rushikonda, Visakhapatnam, Andhra Pradesh 530045',
            'address_te': 'రుషికొండ, విశాఖపట్నం, ఆంధ్ర ప్రదేశ్ 530045',
            'address_hi': 'रुशिकोंडा, विशाखापत्तनम, आंध्र प्रदेश 530045',
            'city': 'Visakhapatnam',
            'state': 'Andhra Pradesh',
            'pincode': '530045',
            'phone': '+91-891-2790101',
            'emergency_phone': '108',
            'specialties': ['Cardiology', 'Neurology', 'Orthopedics', 'Gastroenterology',
                          'Pulmonology', 'Emergency Medicine', 'General Medicine'],
            'services': ['24x7 Emergency', 'ICU', 'NICU', 'Cath Lab', 'Dialysis'],
            'facilities': ['ICU', 'NICU', 'Emergency Ward', 'Modern Equipment', 'Pharmacy'],
            'is_24x7': True,
            'has_emergency': True,
            'has_ambulance': True,
            'bed_capacity': 500,
            'rating': 4.5,
            'total_reviews': 980,
            'verified': True
        },
        {
            'name': 'Seven Hills Hospital',
            'name_te': 'సెవెన్ హిల్స్ ఆస్పత్రి',
            'name_hi': 'सेवन हिल्स अस्पताल',
            'latitude': 17.7317,
            'longitude': 83.3152,
            'address': 'Rockdale Layout, Visakhapatnam, Andhra Pradesh 530002',
            'address_te': 'రాక్‌డేల్ లేఅవుట్, విశాఖపట్నం, ఆంధ్ర ప్రదేశ్ 530002',
            'address_hi': 'रॉकडेल लेआउट, विशाखापत्तनम, आंध्र प्रदेश 530002',
            'city': 'Visakhapatnam',
            'state': 'Andhra Pradesh',
            'pincode': '530002',
            'phone': '+91-891-6671999',
            'emergency_phone': '108',
            'specialties': ['Cardiology', 'Cardiac Surgery', 'Neurology', 'Orthopedics',
                          'Gastroenterology', 'Emergency Medicine'],
            'services': ['24x7 Emergency', 'Heart Surgery', 'Brain Surgery', 'ICU'],
            'facilities': ['State-of-art ICU', 'Cath Lab', 'Modern OT', 'Ambulance Service'],
            'is_24x7': True,
            'has_emergency': True,
            'has_ambulance': True,
            'bed_capacity': 300,
            'rating': 4.6,
            'total_reviews': 756,
            'verified': True
        },
        {
            'name': 'Apollo Hospitals',
            'name_te': 'అపోలో ఆస్పత్రులు',
            'name_hi': 'अपोलो अस्पताल',
            'latitude': 17.7428,
            'longitude': 83.3106,
            'address': 'Waltair Main Road, Visakhapatnam, Andhra Pradesh 530002',
            'address_te': 'వాల్టేర్ మెయిన్ రోడ్, విశాఖపట్నం, ఆంధ్ర ప్రదేశ్ 530002',
            'address_hi': 'वाल्टेयर मेन रोड, विशाखापत्तनम, आंध्र प्रदेश 530002',
            'city': 'Visakhapatnam',
            'state': 'Andhra Pradesh',
            'pincode': '530002',
            'phone': '+91-891-6693333',
            'emergency_phone': '108',
            'specialties': ['Cardiology', 'Neurology', 'Orthopedics', 'Oncology',
                          'Gastroenterology', 'Nephrology', 'Emergency Medicine'],
            'services': ['24x7 Emergency', 'Advanced Diagnostics', 'Robotic Surgery', 'Transplant'],
            'facilities': ['Advanced ICU', 'Cath Lab', 'CT/MRI', 'Blood Bank', 'Pharmacy'],
            'is_24x7': True,
            'has_emergency': True,
            'has_ambulance': True,
            'bed_capacity': 150,
            'rating': 4.7,
            'total_reviews': 1543,
            'verified': True
        },
        {
            'name': 'Care Hospital',
            'name_te': 'కేర్ హాస్పిటల్',
            'name_hi': 'केयर अस्पताल',
            'latitude': 17.7253,
            'longitude': 83.3089,
            'address': 'Asilmetta, Visakhapatnam, Andhra Pradesh 530003',
            'address_te': 'అసిల్మెట్ట, విశాఖపట్నం, ఆంధ్ర ప్రదేశ్ 530003',
            'address_hi': 'असिल्मेट्टा, विशाखापत्तनम, आंध्र प्रदेश 530003',
            'city': 'Visakhapatnam',
            'state': 'Andhra Pradesh',
            'pincode': '530003',
            'phone': '+91-891-6630000',
            'emergency_phone': '108',
            'specialties': ['Cardiology', 'Neurology', 'Orthopedics', 'Gastroenterology',
                          'Pulmonology', 'Emergency Medicine', 'Critical Care'],
            'services': ['24x7 Emergency', 'ICU', 'Trauma Center', 'Diagnostic Center'],
            'facilities': ['Multi-specialty ICU', 'Advanced Diagnostics', 'Modern OT'],
            'is_24x7': True,
            'has_emergency': True,
            'has_ambulance': True,
            'bed_capacity': 200,
            'rating': 4.5,
            'total_reviews': 892,
            'verified': True
        },
        {
            'name': 'Visakha Institute of Medical Sciences',
            'name_te': 'విశాఖ వైద్య శాస్త్ర సంస్థ',
            'name_hi': 'विशाखा चिकित्सा विज्ञान संस्थान',
            'latitude': 17.7386,
            'longitude': 83.3069,
            'address': 'Aganampudi, Visakhapatnam, Andhra Pradesh 530003',
            'address_te': 'అగనంపూడి, విశాఖపట్నం, ఆంధ్ర ప్రదేశ్ 530003',
            'address_hi': 'अगनमपुडी, विशाखापत्तनम, आंध्र प्रदेश 530003',
            'city': 'Visakhapatnam',
            'state': 'Andhra Pradesh',
            'pincode': '530003',
            'phone': '+91-891-2717171',
            'emergency_phone': '108',
            'specialties': ['General Medicine', 'General Surgery', 'Pediatrics', 'Gynecology',
                          'Orthopedics', 'Emergency Medicine'],
            'services': ['24x7 Emergency', 'Maternity Services', 'Pediatric Care'],
            'facilities': ['Emergency Ward', 'Labor Room', 'Pediatric Ward', 'ICU'],
            'is_24x7': True,
            'has_emergency': True,
            'has_ambulance': True,
            'bed_capacity': 250,
            'rating': 4.1,
            'total_reviews': 634,
            'verified': True
        },
        {
            'name': 'Manipal Hospital',
            'name_te': 'మణిపాల్ ఆస్పత్రి',
            'name_hi': 'मणिपाल अस्पताल',
            'latitude': 17.7198,
            'longitude': 83.3025,
            'address': 'Dwarakanagar, Visakhapatnam, Andhra Pradesh 530016',
            'address_te': 'ద్వారకానగర్, విశాఖపట్నం, ఆంధ్ర ప్రదేశ్ 530016',
            'address_hi': 'द्वारकानगर, विशाखापत्तनम, आंध्र प्रदेश 530016',
            'city': 'Visakhapatnam',
            'state': 'Andhra Pradesh',
            'pincode': '530016',
            'phone': '+91-891-3044444',
            'emergency_phone': '108',
            'specialties': ['Cardiology', 'Neurology', 'Orthopedics', 'Oncology',
                          'Nephrology', 'Emergency Medicine', 'Pulmonology'],
            'services': ['24x7 Emergency', 'Cancer Care', 'Kidney Transplant', 'Dialysis'],
            'facilities': ['Advanced ICU', 'Cancer Center', 'Dialysis Unit', 'Blood Bank'],
            'is_24x7': True,
            'has_emergency': True,
            'has_ambulance': True,
            'bed_capacity': 180,
            'rating': 4.6,
            'total_reviews': 1087,
            'verified': True
        },
        {
            'name': 'Queen Mary Hospital',
            'name_te': 'క్వీన్ మేరీ ఆస్పత్రి',
            'name_hi': 'क्वीन मैरी अस्पताल',
            'latitude': 17.7142,
            'longitude': 83.2989,
            'address': 'Jagadamba Junction, Visakhapatnam, Andhra Pradesh 530020',
            'address_te': 'జగదంబ జంక్షన్, విశాఖపట్నం, ఆంధ్ర ప్రదేశ్ 530020',
            'address_hi': 'जगदंबा जंक्शन, विशाखापत्तनम, आंध्र प्रदेश 530020',
            'city': 'Visakhapatnam',
            'state': 'Andhra Pradesh',
            'pincode': '530020',
            'phone': '+91-891-2555555',
            'emergency_phone': '108',
            'specialties': ['General Medicine', 'Pediatrics', 'Gynecology', 'Orthopedics',
                          'Dermatology', 'Emergency Medicine'],
            'services': ['24x7 Emergency', 'Maternity Care', 'Child Care', 'Skin Treatment'],
            'facilities': ['Maternity Ward', 'Pediatric ICU', 'Emergency Ward'],
            'is_24x7': True,
            'has_emergency': True,
            'has_ambulance': True,
            'bed_capacity': 120,
            'rating': 4.0,
            'total_reviews': 456,
            'verified': True
        },
    ]
    
    for hospital_data in hospitals_data:
        hospital = Hospital(**hospital_data)
        db.session.add(hospital)
    
    db.session.commit()
    print(f"✓ {len(hospitals_data)} hospitals initialized")


def initialize_database():
    """Main function to initialize the database"""
    with app.app_context():
        print("Initializing database...")
        
        # Drop all tables and recreate
        db.drop_all()
        db.create_all()
        print("✓ Database tables created")
        
        # Initialize data
        init_specialties()
        init_symptom_categories()
        init_hospitals()
        
        print("\n✅ Database initialization complete!")
        print(f"Total hospitals: {Hospital.query.count()}")
        print(f"Total specialties: {Specialty.query.count()}")
        print(f"Total symptom categories: {SymptomCategory.query.count()}")


if __name__ == '__main__':
    initialize_database()