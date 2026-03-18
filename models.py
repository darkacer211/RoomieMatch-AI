from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    
    # Lifestyle
    cleanliness = db.Column(db.Integer, nullable=False) # 1-5
    noise_tolerance = db.Column(db.Integer, nullable=False) # 1-10
    social_level = db.Column(db.Integer, nullable=False) # 1-10
    guest_freq = db.Column(db.Integer, nullable=False) # 1-5
    home_visit_freq = db.Column(db.Integer, nullable=False) # 0-4
    
    # Hard Constraints (Binary)
    smoking = db.Column(db.Integer, nullable=False) # 0/1
    drinking = db.Column(db.Integer, nullable=False) # 0/1
    utensil_sharing = db.Column(db.Integer, nullable=False) # 0/1
    overnight_guests = db.Column(db.Integer, nullable=False) # 0/1
    pet_friendly = db.Column(db.Integer, nullable=False) # 0/1
    
    # Schedules
    work_shift = db.Column(db.Integer, nullable=False) # 0: Day, 1: Night, 2: Flex
    wfh_days = db.Column(db.Integer, nullable=False) # 0-7
    bathroom_prime = db.Column(db.Integer, nullable=False) # 0: Morning, 1: Evening
    
    # Indian Context
    food_pref = db.Column(db.Integer, nullable=False) # 0: Veg, 1: Non-Veg, 2: Egg
    maid_dependency = db.Column(db.Integer, nullable=False) # 1-5
    ac_usage = db.Column(db.Integer, nullable=False) # 1-5
    
    # Financials
    budget = db.Column(db.Numeric(10, 2), nullable=False)
    bill_punctuality = db.Column(db.Integer, nullable=False) # 1-5

    def __repr__(self):
        return f'<User {self.name}>'
