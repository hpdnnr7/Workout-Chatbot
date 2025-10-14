# FitBot 🏋
FitBot is an interactive fitness chatbot built with Flask, Python, and scikit-learn. It helps users create personalized workout plans based on their fitness goals, equipment availability, and experience level. FitBot can also provide general fitness advice when a specific workout plan isn’t available.

## Features
Responds to user fitness goals and experience level.

Recommends workout plans based on gym access, equipment, or bodyweight exercises.

Provides friendly fallback messages if the input isn’t recognized.

Chat-style interface mimicking iPhone messages:

User messages: blue bubbles

Bot messages: grey bubbles

Centered workout plan boxes for easy readability.

Mobile-friendly layout.

## Example Usage
User: I want to build muscle and I have access to a gym
Bot: Focus on progressive overload and compound lifts like squats, bench, and deadlifts.
Bot (Workout Plan):

<Gym Muscle Builder:
- Bench Press 3x8
- Squats 3x8
- Lat Pulldowns 3x10
- Leg Press 3x10>


If a user asks something without a workout plan (e.g., “mental health”), FitBot will provide a response and suggest trying another fitness-related question.

## Technologies Used

Python 3

Flask – Web framework

scikit-learn – For intent recognition using TF-IDF and cosine similarity

HTML/CSS/JavaScript – Frontend chat interface

## Getting Started

# Prerequisites

Make sure you have Python 3 installed.

pip install flask scikit-learn numpy

## Running Locally

1. Clone the repository:

git clone <your-repo-url>
cd <your-repo-folder>


2. Run the Flask app:

python main.py


3. Open your browser or mobile device and go to:

http://localhost:81

## File Structure
.
├── main.py           # Flask app and chatbot logic
├── templates/
│   └── index.html    # Frontend chat interface
└── README.md         # Project documentation
