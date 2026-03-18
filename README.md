# RoomieMatch AI

RoomieMatch AI is a machine-learning-powered roommate matching platform. It evaluates a comprehensive set of lifestyle and preference parameters (such as cleanliness, noise tolerance, smoking/drinking habits, rent budget, etc.) to pair potential roommates intelligently.

## Features

- **Profile System:** Comprehensive user surveys storing preferences and habits.
- **Hybrid Matching Engine:** Employs Cosine Similarity alongside a robust Random Forest Regressor to match profiles.
- **Dealbreaker Logic:** Severe penalties for absolute mismatches (e.g. food preferences, smoking).
- **Interactive UI:** Visually compares profiles listing key Green Flags and Red Flags.

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Initial Model:**
   ```bash
   python data_engine.py
   ```

3. **Run the Application:**
   ```bash
   python app.py
   ```
   *The app uses a local SQLite database that initializes automatically.*

## License
MIT
