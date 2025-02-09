from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

keyword_data = {
    "resume": "A document that presents a person's background, skills, and accomplishments.",
    "cv": "A comprehensive document detailing academic and professional history.",
    "cover letter": "A letter sent with a resume to introduce yourself and highlight your qualifications.",
    "skills": "Abilities and expertise that are relevant to a job or role."
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get("query", "").lower()
    result = keyword_data.get(query, "No explanation found. Try a different keyword.")
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)