from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

intents = [
    {
        "intent": "lose_weight_no_equipment",
        "text": "I want to lose weight but I don't have any equipment"
    },
    {
        "intent": "build_muscle_gym",
        "text": "I want to build muscle and I have access to a gym"
    },
    {
        "intent": "stay_fit_dumbbells",
        "text": "I just want to stay fit and I have dumbbells"
    },
    {
        "intent": "beginner_bodyweight",
        "text": "I'm a beginner with no equipment"
    },
    {
        "intent": "advanced_dumbbell",
        "text": "I'm advanced and have dumbbells"
    },
    {
        "intent": "intermediate_gym",
        "text": "I'm intermediate and want to work out at the gym"
    },
    {
        "intent": "nutrition_help",
        "text": "I need help with what to eat to lose weight"
    },
    {
        "intent": "postpartum_fitness",
        "text": "I'm a new mom looking to get back in shape"
    },
    {
        "intent": "injury_recovery",
        "text": "I'm recovering from an injury and want to ease into fitness"
    },
    {
        "intent": "cardio_focus",
        "text": "I want to improve my cardio health"
    },
    {
        "intent": "mental_health_focus",
        "text": "I'm working out to improve my mental health"
    },
]

intent_responses = {
    "lose_weight_no_equipment":
    "Try bodyweight HIIT workouts at home. Keep moving, even without equipment!",
    "build_muscle_gym":
    "Focus on progressive overload and compound lifts like squats, bench, and deadlifts.",
    "stay_fit_dumbbells":
    "Incorporate full-body circuits using dumbbells 3–4 times per week.",
    "beginner_bodyweight":
    "Start with 3 rounds of squats, push-ups, and planks. Consistency > intensity.",
    "advanced_dumbbell":
    "Push yourself with supersets and heavier weights — focus on time under tension.",
    "intermediate_gym":
    "Mix cardio with strength training to avoid plateaus.",
    "nutrition_help":
    "Aim for high protein, moderate carbs, and plenty of vegetables. Track your intake.",
    "postpartum_fitness":
    "Start slow with core and pelvic floor work. Always get clearance from your doctor first.",
    "injury_recovery":
    "Prioritize mobility, stretching, and low-impact exercises. Listen to your body!",
    "cardio_focus":
    "Try steady-state cardio like walking or jogging 30 mins a day, 5x/week.",
    "mental_health_focus":
    "Exercise helps a ton with mood. Choose movement you enjoy — even dancing counts!",
}

workout_plans = {
    "lose_weight_no_equipment":
    "30-Minute Fat Burn:\n- 3x15 Jumping Jacks\n- 3x10 Burpees\n- 3x20 High Knees\n- 3x30s Mountain Climbers",
    "build_muscle_gym":
    "Gym Muscle Builder:\n- Bench Press 3x8\n- Squats 3x8\n- Lat Pulldowns 3x10\n- Leg Press 3x10",
    "stay_fit_dumbbells":
    "Dumbbell Total Body:\n- Goblet Squats 3x12\n- Dumbbell Rows 3x10\n- Overhead Press 3x10\n- Deadlifts 3x8",
    "beginner_bodyweight":
    "Beginner Bodyweight:\n- Wall Sits 3x30s\n- Modified Pushups 3x10\n- Step Ups 3x10 each leg\n- Glute Bridges 3x15",
    "advanced_dumbbell":
    "Advanced Dumbbell:\n- Bulgarian Split Squats 3x10\n- Renegade Rows 3x8\n- Dumbbell Snatch 3x6\n- Thrusters 3x10",
    "intermediate_gym":
    "Intermediate Gym:\n- Incline Bench 3x10\n- Barbell Rows 3x8\n- Walking Lunges 3x12\n- Cable Pushdowns 3x15",
}

fallback_message = (
    "<div class='fallback-message'>"
    "Hmm, I’m not sure how to help with that yet.<br><br>"
    "Here are some things you can ask me:"
    "<ul class='fallback-list'>"
    "<li>I want to lose weight but I don't have any equipment</li>"
    "<li>I'm a beginner with no equipment</li>"
    "<li>I'm advanced and have dumbbells</li>"
    "<li>I want to build muscle and I have access to a gym</li>"
    "<li>I just want to stay fit and I have dumbbells</li>"
    "</ul>"
    "</div>")

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
intent_texts = [i["text"] for i in intents]
X = vectorizer.fit_transform(intent_texts)


def get_intent(user_input, threshold=0.3):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    best_index = similarity.argmax()
    confidence = similarity[0, best_index]
    if confidence < threshold:
        return None, confidence
    return intents[best_index]["intent"], confidence


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    # Get JSON data from request
    data = request.get_json()
    print("Received data:", data)

    user_input = data.get("user_input", "") if data else ""
    print("User input:", user_input)  # Check the input in console

    intent, confidence = get_intent(user_input)

    if intent is None:
        response = fallback_message
    else:
        response = intent_responses.get(
            intent, "I'm not sure how to help with that yet.")
        plan = workout_plans.get(intent, None)
        if plan:
            plan_html = plan.replace("\n", "<br>")
            response += (
                "<br><br>"
                "<div class='workout-box'>"
                "Here's a workout plan I recommend:<br><br>"
                f"{plan_html}"
                "</div>"
            )

    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81)
