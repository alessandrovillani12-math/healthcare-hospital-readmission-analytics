import pandas as pd
import numpy as np

# Set seed to ensure reproducible random data generation
np.random.seed(42)

# Set number of pazients
n_patients = 200

# =====================================================================
# 1. GEN. DIM_PAZIENTS (Anagrafic)
# =====================================================================
patient_ids = [f"P{str(i).zfill(4)}" for i in range(1, n_patients + 1)]
#Generate a random group of patients with these characteristics
ages = np.random.randint(30, 85, size=n_patients) 
genres = np.random.choice(['M', 'F'], size=n_patients, p=[0.48, 0.52])
ethnicities = np.random.choice(['Caucasian', 'African American', 'Asian', 'Hispanic'], size=n_patients, p=[0.70, 0.15, 0.10, 0.05])

df_patients = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Age': ages,
    'Gender': genres,
    'Ethnicity': ethnicities
})

# =====================================================================
# 2. GENERATION: DIM_LIFESTYLE (Behavioral Factors)
# =====================================================================
smokers = np.random.choice([1, 0], size=n_patients, p=[0.22, 0.78])
activity = np.random.choice(['High', 'Medium', 'Low'], size=n_patients, p=[0.25, 0.45, 0.30])
alcohol = np.random.choice(['Non-consumer', 'Moderate', 'Excessive'], size=n_patients, p=[0.40, 0.50, 0.10])

df_lifestyle = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Smoker': smokers,
    'Activity': activity,
    'Alcohol': alcohol
})

# =====================================================================
# 3. GENERATION: FACT_CLINICAL_VISITS (Transactional Data with Correlation)
# =====================================================================
visits_data = []
visit_id_counter = 1

for idx, row in df_patients.iterrows():
    p_id = row['Patient_ID']
    age = row['Age']
    
    # Retrieve the patient's lifestyle to calculate correlated clinical parameters
    p_smoke = df_lifestyle.loc[df_lifestyle['Patient_ID'] == p_id, 'Smoker'].values[0]
    p_act = df_lifestyle.loc[df_lifestyle['Patient_ID'] == p_id, 'Physical_Activity'].values[0]
    
    # Each patient makes 1 to 3 visits over time (Simulating clinical history)
    num_visits = np.random.randint(1, 4)
    
    for v in range(num_visits):
        # --- MATHEMATICAL MODELING OF CLINICAL PARAMETERS ---
        # Systolic blood pressure increases with age, smoking, and sedentary lifestyle + a Gaussian noise component
        base_sys = 115 + (age * 0.25) + (p_smoke * 8) + (10 if p_act == 'Low' else 0)
        systolic_pressure = int(np.random.normal(base_sys, 7))
        
        # Diastolic pressure is linked to systolic pressure
        diastolic_pressure = int(systolic_pressure * 0.65 + np.random.normal(0, 4))
        
        # Total cholesterol influenced by age and smoking
        base_chol = 180 + (age * 0.4) + (p_smoke * 15)
        cholesterol = int(np.random.normal(base_chol, 20))
        
        # Blood glucose influenced by age and low physical activity
        base_gluc = 85 + (age * 0.2) + (15 if p_act == 'Low' else 0)
        glucose = int(np.random.normal(base_gluc, 12))
        
        # BMI (Body Mass Index)
        bmi = round(np.random.normal(26, 4) + (2 if p_act == 'Low' else -1), 1)
        
        # --- REAL RISK CALCULATION (Simplified Logistic Model) ---
        # Assign mathematical weights to risk factors to determine if a cardiovascular event occurs
        risk_score = (age * 0.04) + (p_smoke * 0.8) + (0.02 * (systolic_pressure - 120)) + (0.01 * (cholesterol - 200)) - (0.5 if p_act == 'High' else 0)
        
        # Sigmoid function to map the score into a probability range [0, 1]
        event_prob = 1 / (1 + np.exp(-(risk_score - 3.5)))
        cardio_event = 1 if np.random.rand() < event_prob else 0
        
        # Random date within recent years
        year = np.random.choice([2024, 2025, 2026])
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 28)
        visit_date = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
        
        visits_data.append({
            'Visit_ID': f"V{str(visit_id_counter).zfill(5)}",
            'Patient_ID': p_id,
            'Visit_Date': visit_date,
            'Systolic_Pressure': systolic_pressure,
            'Diastolic_Pressure': diastolic_pressure,
            'Total_Cholesterol': cholesterol,
            'Fasting_Glucose': glucose,
            'BMI': bmi,
            'Cardiovascular_Event': cardio_event
        })
        visit_id_counter += 1

df_visits = pd.DataFrame(visits_data)

# =====================================================================
# 4. EXPORTING FILES TO CSV
# =====================================================================
df_patients.to_csv('data/Dim_Patients.csv', index=False)
df_lifestyle.to_csv('data/Dim_Lifestyle.csv', index=False)
df_visits.to_csv('data/Fact_Clinical_Visits.csv', index=False)

print("Pipeline completed successfully! The 3 Star Schema structured datasets are ready in 'data/'."
