import random
from app import app, db
from models import User

def seed():
    names = ['Ketan', 'Yash', 'Krushna', 'Sam', 'Atharwa']
    with app.app_context():
        for name in names:
            user = User(
                name=name,
                cleanliness=random.randint(1, 5),
                noise_tolerance=random.randint(1, 10),
                social_level=random.randint(1, 10),
                guest_freq=random.randint(1, 5),
                home_visit_freq=random.randint(0, 4),
                smoking=random.randint(0, 1),
                drinking=random.randint(0, 1),
                utensil_sharing=random.randint(0, 1),
                overnight_guests=random.randint(0, 1),
                pet_friendly=random.randint(0, 1),
                work_shift=random.randint(0, 2),
                wfh_days=random.randint(0, 7),
                bathroom_prime=random.randint(0, 1),
                food_pref=random.randint(0, 2),
                maid_dependency=random.randint(1, 5),
                ac_usage=random.randint(1, 5),
                budget=round(random.uniform(500, 3000), 2),
                bill_punctuality=random.randint(1, 5)
            )
            db.session.add(user)
        db.session.commit()
    print("Test users seeded successfully.")

if __name__ == '__main__':
    seed()
