import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import random
import os

def generate_synthetic_data(num_pairs=10000):
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    for _ in range(num_pairs):
        # User 1
        u1 = {
            'cleanliness': random.randint(1, 5),
            'noise_tolerance': random.randint(1, 10),
            'social_level': random.randint(1, 10),
            'guest_freq': random.randint(1, 5),
            'home_visit_freq': random.randint(0, 4),
            'smoking': random.randint(0, 1),
            'drinking': random.randint(0, 1),
            'utensil_sharing': random.randint(0, 1),
            'overnight_guests': random.randint(0, 1),
            'pet_friendly': random.randint(0, 1),
            'work_shift': random.randint(0, 2),
            'wfh_days': random.randint(0, 7),
            'bathroom_prime': random.randint(0, 1),
            'food_pref': random.randint(0, 2),
            'maid_dependency': random.randint(1, 5),
            'ac_usage': random.randint(1, 5),
            'budget': round(random.uniform(500, 3000), 2),
            'bill_punctuality': random.randint(1, 5)
        }
        
        # User 2
        u2 = {
            'cleanliness': random.randint(1, 5),
            'noise_tolerance': random.randint(1, 10),
            'social_level': random.randint(1, 10),
            'guest_freq': random.randint(1, 5),
            'home_visit_freq': random.randint(0, 4),
            'smoking': random.randint(0, 1),
            'drinking': random.randint(0, 1),
            'utensil_sharing': random.randint(0, 1),
            'overnight_guests': random.randint(0, 1),
            'pet_friendly': random.randint(0, 1),
            'work_shift': random.randint(0, 2),
            'wfh_days': random.randint(0, 7),
            'bathroom_prime': random.randint(0, 1),
            'food_pref': random.randint(0, 2),
            'maid_dependency': random.randint(1, 5),
            'ac_usage': random.randint(1, 5),
            'budget': round(random.uniform(500, 3000), 2),
            'bill_punctuality': random.randint(1, 5)
        }
        
        # Step A (Cosine Similarity on all numeric/normalized features)
        # We normalize budget by 3000 to keep it in a small range
        vec1 = np.array([[
            u1['cleanliness'], u1['noise_tolerance'], u1['social_level'], u1['guest_freq'], u1['home_visit_freq'],
            u1['smoking'], u1['drinking'], u1['utensil_sharing'], u1['overnight_guests'], u1['pet_friendly'],
            u1['work_shift'], u1['wfh_days'], u1['bathroom_prime'], u1['food_pref'], u1['maid_dependency'],
            u1['ac_usage'], u1['budget']/3000, u1['bill_punctuality']
        ]])
        vec2 = np.array([[
            u2['cleanliness'], u2['noise_tolerance'], u2['social_level'], u2['guest_freq'], u2['home_visit_freq'],
            u2['smoking'], u2['drinking'], u2['utensil_sharing'], u2['overnight_guests'], u2['pet_friendly'],
            u2['work_shift'], u2['wfh_days'], u2['bathroom_prime'], u2['food_pref'], u2['maid_dependency'],
            u2['ac_usage'], u2['budget']/3000, u2['bill_punctuality']
        ]])
        
        cos_sim = cosine_similarity(vec1, vec2)[0][0]
        
        # Step B (Delta Features)
        # Absolute differences for numerics
        delta_cleanliness = abs(u1['cleanliness'] - u2['cleanliness'])
        delta_noise = abs(u1['noise_tolerance'] - u2['noise_tolerance'])
        delta_social = abs(u1['social_level'] - u2['social_level'])
        delta_guest_freq = abs(u1['guest_freq'] - u2['guest_freq'])
        delta_home_visit = abs(u1['home_visit_freq'] - u2['home_visit_freq'])
        delta_wfh = abs(u1['wfh_days'] - u2['wfh_days'])
        delta_maid = abs(u1['maid_dependency'] - u2['maid_dependency'])
        delta_ac = abs(u1['ac_usage'] - u2['ac_usage'])
        delta_budget = float(abs(u1['budget'] - u2['budget']))
        delta_punctuality = abs(u1['bill_punctuality'] - u2['bill_punctuality'])
        
        # XOR gates / Mismatches for binary and categoricals
        mismatch_smoking = 1 if u1['smoking'] != u2['smoking'] else 0
        mismatch_drinking = 1 if u1['drinking'] != u2['drinking'] else 0
        mismatch_utensils = 1 if u1['utensil_sharing'] != u2['utensil_sharing'] else 0
        mismatch_overnight = 1 if u1['overnight_guests'] != u2['overnight_guests'] else 0
        mismatch_pets = 1 if u1['pet_friendly'] != u2['pet_friendly'] else 0
        
        mismatch_work_shift = 1 if u1['work_shift'] != u2['work_shift'] else 0
        mismatch_bathroom = 1 if u1['bathroom_prime'] != u2['bathroom_prime'] else 0
        mismatch_food = 1 if u1['food_pref'] != u2['food_pref'] else 0
        
        # Step C: Ground truth calculation (Heavy penalty on Hard Constraints)
        base_score = 100
        
        # Major dealbreakers - Exaggerated for final model
        base_score -= mismatch_smoking * 40
        base_score -= mismatch_drinking * 30
        base_score -= mismatch_food * 40 # Food mismatch is heavy in Indian context
        base_score -= mismatch_pets * 30
        base_score -= mismatch_overnight * 20
        base_score -= mismatch_utensils * 15

        # Numeric deductions
        base_score -= (delta_cleanliness / 4) * 15
        base_score -= (delta_noise / 9) * 10
        base_score -= (delta_social / 9) * 10
        base_score -= (delta_budget / 2500) * 15
        base_score -= (delta_ac / 4) * 8
        base_score -= (delta_maid / 4) * 5
        base_score -= mismatch_work_shift * 5
        base_score -= mismatch_bathroom * 5
        base_score -= (delta_punctuality / 4) * 10
        
        # Cosine sim bonus (up to 5 points)
        score = base_score + (cos_sim * 5)
        
        # Bounds check
        score = max(0, min(100, score))
        
        data.append({
            'delta_cleanliness': delta_cleanliness,
            'delta_noise': delta_noise,
            'delta_social': delta_social,
            'delta_guest_freq': delta_guest_freq,
            'delta_home_visit': delta_home_visit,
            'delta_wfh': delta_wfh,
            'delta_maid': delta_maid,
            'delta_ac': delta_ac,
            'delta_budget': delta_budget,
            'delta_punctuality': delta_punctuality,
            'mismatch_smoking': mismatch_smoking,
            'mismatch_drinking': mismatch_drinking,
            'mismatch_utensils': mismatch_utensils,
            'mismatch_overnight': mismatch_overnight,
            'mismatch_pets': mismatch_pets,
            'mismatch_work_shift': mismatch_work_shift,
            'mismatch_bathroom': mismatch_bathroom,
            'mismatch_food': mismatch_food,
            'cosine_sim': cos_sim,
            'score': score
        })
        
    return pd.DataFrame(data)

def train_and_save_model():
    print("Generating 10,000 synthetic data pairs for V2...")
    df = generate_synthetic_data(10000)
    
    X = df.drop('score', axis=1)
    y = df['score']
    
    print("Training Hybrid Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print("Saving model to model.pkl...")
    joblib.dump(model, 'model.pkl')
    
    # Save feature columns to ensure alignment mapping during inference
    joblib.dump(list(X.columns), 'model_features.pkl')
    print("Done!")

if __name__ == "__main__":
    train_and_save_model()
