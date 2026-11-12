from flask import Flask, request, jsonify, render_template
from model_loader import df_model, sentence_model
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches


app = Flask(__name__)
gemini_model = genai.GenerativeModel('gemini-pro')

# Expanded Greetings
GREETINGS = {
    'hello': "Kuzuzangpola! I'm your Bhutan travel assistant. How can I help you explore the Land of the Thunder Dragon today?",
    'hi': "Kuzu zangpo! Welcome to your Bhutan travel guide. What would you like to know about this magical kingdom?",
    'how are you': "I'm doing wonderfully, thank you! Ready to help you discover Bhutan's hidden treasures. Where shall we start?",
    'hey': "Kuzu zangpo-la! Excited to help you plan your Bhutan adventure. What's on your travel wishlist?",
    'good morning': "Good morning! Ready for a fresh start exploring Bhutan?",
    'good afternoon': "Good afternoon! Looking for travel ideas in Bhutan?",
    'good evening': "Good evening! Let’s explore Bhutan together!",
    'kuzu zangpo': "Kuzu zangpo la! How can I assist you on your Bhutan journey?"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data['query'].strip().lower()
    
    # 1. Check for greetings
    close_matches = get_close_matches(query, GREETINGS.keys(), n=1, cutoff=0.6)
    if close_matches:
        matched = close_matches[0]
        return jsonify({
            'response': GREETINGS[matched],
            'suggestions': []
        })


    # 2. Handle direct questions about location, climate, or budget
    location_response = handle_location_query(query)
    if location_response:
        return jsonify({
            'response': location_response,
            'suggestions': []
        })

    # 3. Check for follow-up questions
    if any(phrase in query for phrase in ['tell me about', 'details about', 'give me more information', 'more about']):
        place = extract_place_name(query)
        response = handle_followup(place, df_model)
        return jsonify({
            'response': response,
            'suggestions': []
        })

    # 4. Process travel queries using semantic similarity
    query_embedding = sentence_model.encode(query)
    doc_embeddings = sentence_model.encode(df_model['combined_info'].tolist())
    df_model['similarity'] = cosine_similarity([query_embedding], doc_embeddings)[0]

    results = df_model.sort_values('similarity', ascending=False).head(3)
    response = format_response(results)

    return jsonify({
        'response': response,
        'suggestions': results['place_name'].tolist()
    })

def extract_place_name(query):
    for phrase in ['tell me about', 'where is ','details about', 'more about']:
        if phrase in query:
            return query.split(phrase)[-1].strip()
    return query

def handle_followup(place, df):
    place_lower = place.lower()
    for _, row in df.iterrows():
        if place_lower in row['place_name'].lower():
            return (
                f" <strong>Detailed Information About {row['place_name']}</strong><br><br>"
                f" <strong>Location:</strong> {row['region']} region<br>"
                f" <strong>Description:</strong> {row['descriptions']}<br>"
                f" <strong>Budget:</strong> {row['budget']}<br>"
                f" <strong>Climate:</strong> {row['climate']}<br>"
                f" <strong>Type:</strong> {row['location_type']}<br><br>"
                f" <strong>Travel Tip:</strong> {get_travel_tip(row)}<br>"
                f"🔗 <a href='{row['url']}' target='_blank'>Click here to learn more</a>"
            )
    return "Sorry, I couldn't find details about that location."

def get_travel_tip(row):
    tips = {
        'valley': "Best visited in the morning for clear views.",
        'monastery': "Remember to walk clockwise around stupas.",
        'village': "Try local homemade butter tea.",
        'glacier': "Wear proper hiking boots.",
        'city': "Visit the weekend market for souvenirs."
    }
    tip = tips.get(row['location_type'].lower(), "Check opening hours before visiting.")
    return f"{tip} <a href='{row['url']}' target='_blank'>Learn more</a>"

def handle_location_query(query):
    query = query.lower()

    for _, row in df_model.iterrows():
        place = row['place_name'].lower()
        if place in query:
            if any(word in query for word in ["where is", "where's", "located", "location of"]):
                return (
                    f"{row['place_name']} is located in the {row['region']} region of Bhutan. <br><br>"
                    f" {row['descriptions']}<br>"
                    f" <a href='{row['url']}' target='_blank'>Click here for more info</a>"
                )
            elif any(word in query for word in ["climate", "weather", "temperature", "cold", "hot"]):
                return f"The climate in {row['place_name']} is {row['climate'].lower()}. 🔗 <a href='{row['url']}' target='_blank'>More info</a>"
            elif any(word in query for word in ["budget", "cost", "price", "expense", "affordable"]):
                return f"The budget for visiting {row['place_name']} is considered {row['budget'].lower()}. 🔗 <a href='{row['url']}' target='_blank'>More info</a>"
    return None

def format_response(results):
    if len(results) == 0:
        return "I couldn't find matching destinations. Could you try different wording?"
    
    response = "🏔 <strong>Top Recommendations:</strong><br><br>"
    for _, row in results.iterrows():
        response += (
            f" <strong>{row['place_name']} ({row['region']})</strong><br>"
            f"• {row['descriptions']}<br>"
            f"• Budget: {row['budget']} | Climate: {row['climate']}<br>"
            f"🔗 <a href='{row['url']}' target='_blank'>Visit here</a><br><br>"
        )
    response += "Ask me for more details about any of these places!"
    return response


if __name__ == '__main__':
    app.run(debug=True)
