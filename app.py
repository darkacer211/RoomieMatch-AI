from flask import Flask, render_template, request, redirect, url_for, flash
from config import Config
from models import db, User
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

# Load model and feature names
MODEL_PATH = 'model.pkl'
FEATURES_PATH = 'model_features.pkl'
model = None
feature_names = None
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
except Exception as e:
    print(f"Failed to load model or features: {e}")

def get_feature_icon(feat):
    mapping = {
        'delta_cleanliness': 'fa-broom',
        'delta_noise': 'fa-volume-up',
        'delta_social': 'fa-users',
        'delta_guest_freq': 'fa-user-friends',
        'delta_home_visit': 'fa-home',
        'delta_wfh': 'fa-laptop-house',
        'delta_maid': 'fa-pump-soap',
        'delta_ac': 'fa-snowflake',
        'delta_budget': 'fa-wallet',
        'delta_punctuality': 'fa-clock',
        'mismatch_smoking': 'fa-smoking',
        'mismatch_drinking': 'fa-glass-martini-alt',
        'mismatch_utensils': 'fa-utensils',
        'mismatch_overnight': 'fa-bed',
        'mismatch_pets': 'fa-paw',
        'mismatch_work_shift': 'fa-briefcase',
        'mismatch_bathroom': 'fa-bath',
        'mismatch_food': 'fa-hamburger'
    }
    return mapping.get(feat, 'fa-circle-notch')

def predict_compatibility(u1, u2):
    if model is None or feature_names is None:
        return None, "Model not loaded"
        
    vec1 = np.array([[
        u1.cleanliness, u1.noise_tolerance, u1.social_level, u1.guest_freq, u1.home_visit_freq,
        u1.smoking, u1.drinking, u1.utensil_sharing, u1.overnight_guests, u1.pet_friendly,
        u1.work_shift, u1.wfh_days, u1.bathroom_prime, u1.food_pref, u1.maid_dependency,
        u1.ac_usage, float(u1.budget)/3000, u1.bill_punctuality
    ]])
    vec2 = np.array([[
        u2.cleanliness, u2.noise_tolerance, u2.social_level, u2.guest_freq, u2.home_visit_freq,
        u2.smoking, u2.drinking, u2.utensil_sharing, u2.overnight_guests, u2.pet_friendly,
        u2.work_shift, u2.wfh_days, u2.bathroom_prime, u2.food_pref, u2.maid_dependency,
        u2.ac_usage, float(u2.budget)/3000, u2.bill_punctuality
    ]])
    
    cos_sim = cosine_similarity(vec1, vec2)[0][0]
    
    input_data = {
        'delta_cleanliness': abs(u1.cleanliness - u2.cleanliness),
        'delta_noise': abs(u1.noise_tolerance - u2.noise_tolerance),
        'delta_social': abs(u1.social_level - u2.social_level),
        'delta_guest_freq': abs(u1.guest_freq - u2.guest_freq),
        'delta_home_visit': abs(u1.home_visit_freq - u2.home_visit_freq),
        'delta_wfh': abs(u1.wfh_days - u2.wfh_days),
        'delta_maid': abs(u1.maid_dependency - u2.maid_dependency),
        'delta_ac': abs(u1.ac_usage - u2.ac_usage),
        'delta_budget': float(abs(u1.budget - u2.budget)),
        'delta_punctuality': abs(u1.bill_punctuality - u2.bill_punctuality),
        
        'mismatch_smoking': 1 if u1.smoking != u2.smoking else 0,
        'mismatch_drinking': 1 if u1.drinking != u2.drinking else 0,
        'mismatch_utensils': 1 if u1.utensil_sharing != u2.utensil_sharing else 0,
        'mismatch_overnight': 1 if u1.overnight_guests != u2.overnight_guests else 0,
        'mismatch_pets': 1 if u1.pet_friendly != u2.pet_friendly else 0,
        
        'mismatch_work_shift': 1 if u1.work_shift != u2.work_shift else 0,
        'mismatch_bathroom': 1 if u1.bathroom_prime != u2.bathroom_prime else 0,
        'mismatch_food': 1 if u1.food_pref != u2.food_pref else 0,
        
        'cosine_sim': cos_sim
    }
    
    # Ensure correct order
    features_df = pd.DataFrame([input_data]).reindex(columns=feature_names)
    score = model.predict(features_df)[0]
    
    green_flags = []
    red_flags = []
    
    importances = model.feature_importances_
    
    for i, col in enumerate(feature_names):
        if col != 'cosine_sim':
            val = input_data[col]
            norm = 1.0
            if 'budget' in col: norm = 2500
            elif 'cleanliness' in col or 'maid' in col or 'ac' in col or 'guest' in col or 'home_visit' in col or 'punctuality' in col: norm = 4.0
            elif 'noise' in col or 'social' in col: norm = 9.0
            elif 'wfh' in col: norm = 7.0
            
            norm_val = val / norm
            impact = norm_val * importances[i]
            
            # Use original display mapping for text
            mapping = {
                'delta_cleanliness': 'Cleanliness', 'delta_noise': 'Noise Tolerance', 'delta_social': 'Social Level',
                'delta_guest_freq': 'Guest Frequency', 'delta_home_visit': 'Home Visit Frequency', 'delta_wfh': 'WFH Schedule',
                'delta_maid': 'Maid Dependency', 'delta_ac': 'AC Usage', 'delta_budget': 'Budget', 'delta_punctuality': 'Bill Punctuality',
                'mismatch_smoking': 'Smoking Habit', 'mismatch_drinking': 'Drinking Habit', 'mismatch_utensils': 'Utensil Sharing',
                'mismatch_overnight': 'Overnight Guests', 'mismatch_pets': 'Pet Friendliness', 'mismatch_work_shift': 'Work Shift',
                'mismatch_bathroom': 'Bathroom Routine', 'mismatch_food': 'Dietary Preference'
            }
            feat_name = mapping.get(col, col)
            icon = get_feature_icon(col)
            
            if val == 0:
                green_flags.append({'name': f"Perfect match on {feat_name}", 'icon': icon, 'weight': importances[i]})
            elif impact > 0.025 or ('mismatch' in col and val > 0): 
                red_flags.append({'name': f"Mismatch on {feat_name}", 'icon': icon, 'weight': impact})
                
    # Sort flags by importance/impact
    green_flags.sort(key=lambda x: x['weight'], reverse=True)
    red_flags.sort(key=lambda x: x['weight'], reverse=True)
    
    return score, green_flags[:5], red_flags[:5]

def get_category(score):
    if score >= 80:
        return "High Compatibility"
    elif score >= 60:
        return "Moderate Compatibility"
    elif score >= 40:
        return "Low Compatibility"
    else:
        return "Incompatible"

@app.route('/')
def index():
    users = User.query.all()
    return render_template('index.html', users=users)

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        user = User(
            name=request.form.get('name'),
            cleanliness=int(request.form.get('cleanliness')),
            noise_tolerance=int(request.form.get('noise_tolerance')),
            social_level=int(request.form.get('social_level')),
            guest_freq=int(request.form.get('guest_freq')),
            home_visit_freq=int(request.form.get('home_visit_freq')),
            
            smoking=int(request.form.get('smoking')),
            drinking=int(request.form.get('drinking')),
            utensil_sharing=int(request.form.get('utensil_sharing')),
            overnight_guests=int(request.form.get('overnight_guests')),
            pet_friendly=int(request.form.get('pet_friendly')),
            
            work_shift=int(request.form.get('work_shift')),
            wfh_days=int(request.form.get('wfh_days')),
            bathroom_prime=int(request.form.get('bathroom_prime')),
            
            food_pref=int(request.form.get('food_pref')),
            maid_dependency=int(request.form.get('maid_dependency')),
            ac_usage=int(request.form.get('ac_usage')),
            
            budget=float(request.form.get('budget')),
            bill_punctuality=int(request.form.get('bill_punctuality'))
        )
        db.session.add(user)
        db.session.commit()
        flash('Profile saved successfully!', 'success')
        return redirect(url_for('index'))
        
    return render_template('survey.html')

@app.route('/auto_match/<int:user_id>')
def auto_match(user_id):
    target_user = db.session.get(User, user_id)
    if not target_user:
        flash('User not found.', 'danger')
        return redirect(url_for('index'))
        
    if model is None:
        flash('Machine learning model not found. Please train it first.', 'warning')
        return redirect(url_for('index'))
        
    other_users = User.query.filter(User.id != user_id).all()
    if not other_users:
        flash('No other profiles found in the database to match with!', 'warning')
        return redirect(url_for('index'))
        
    best_score = -1
    best_match = None
    best_explanation = ""
    
    for potential_match in other_users:
        score, g_flags, r_flags = predict_compatibility(target_user, potential_match)
        if score > best_score:
            best_score = score
            best_match = potential_match
            best_green_flags = g_flags
            best_red_flags = r_flags
            
    category = get_category(best_score)
    
    return render_template('compare.html', 
                           user1=target_user, 
                           user2=best_match, 
                           score=best_score, 
                           category=category, 
                           green_flags=best_green_flags,
                           red_flags=best_red_flags,
                           is_auto_match=True)

@app.route('/compare/<int:id1>/<int:id2>')
def compare(id1, id2):
    user1 = db.session.get(User, id1)
    user2 = db.session.get(User, id2)
    
    if not user1 or not user2:
        flash('One or both users not found.', 'danger')
        return redirect(url_for('index'))
        
    if model is None:
        flash('Machine learning model not found. Please train it first.', 'warning')
        return redirect(url_for('index'))
        
    score, green_flags, red_flags = predict_compatibility(user1, user2)
    category = get_category(score)
    
    return render_template('compare.html', user1=user1, user2=user2, score=score, category=category, green_flags=green_flags, red_flags=red_flags)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
