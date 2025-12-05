from flask import Flask, request, jsonify
from services.crop_service import predict_crop_choice

app = Flask(__name__)

@app.route("/")
def home():
    return "Agri AI Backend Running 🧠🌾"

@app.route("/api/crop/check", methods=["POST"])
def check_crop():
    data = request.get_json()

    result = predict_crop_choice(data)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
