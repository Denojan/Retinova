from flask import Blueprint, request, jsonify
from .services.model_service import predict_image_amd

api_bp = Blueprint('api', __name__)

@api_bp.route("/", methods=["GET"])
def index():
    return "Flask backend is up and running!"

# @api_bp.route("/predict", methods=["POST"])
# def predict_route():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No file selected"}), 400

#     try:
#         prediction = predict_image(file)
#         return jsonify(prediction)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


@api_bp.route("/predict/amd", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        prediction = predict_image_amd(file)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500