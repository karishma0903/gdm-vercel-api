from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load models
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "..", "models")

def model_path(file):
    return os.path.join(model_dir, file)

with open(model_path("classification_pca.pkl"), "rb") as f:
    classification_pca = pickle.load(f)

with open(model_path("classification_student_rf.pkl"), "rb") as f:
    classification_model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array([
            data["PPBS"], data["GCT"], data["Height"], data["Weight of baby"],
            data["BP-DIASTOLE"], data["TSH"], data["FT4"]
        ]).reshape(1, -1)

        features_pca = classification_pca.transform(features)
        gdm_pred = classification_model.predict(features_pca)[0]
        gdm_pred_class = int(round(gdm_pred))

        gdm_type = "GDM Type I" if gdm_pred_class == 0 else "GDM Type II"

        return jsonify({"gdm_type": gdm_type})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
