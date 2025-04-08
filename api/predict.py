import json
import pickle
import numpy as np
import os

# Load models
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def model_path(file): return os.path.join(base_dir, "models", file)

with open(model_path("classification_pca.pkl"), "rb") as f:
    classification_pca = pickle.load(f)

with open(model_path("classification_student_rf.pkl"), "rb") as f:
    classification_model = pickle.load(f)

def handler(request):
    if request.method != "POST":
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Only POST method is allowed"})
        }

    try:
        data = request.json
        features = np.array([
            data["PPBS"], data["GCT"], data["Height"], data["Weight of baby"],
            data["BP-DIASTOLE"], data["TSH"], data["FT4"]
        ]).reshape(1, -1)

        features_pca = classification_pca.transform(features)
        gdm_pred = classification_model.predict(features_pca)[0]
        gdm_pred_class = int(round(gdm_pred))

        gdm_type = "GDM Type I" if gdm_pred_class == 0 else "GDM Type II"

        return {
            "statusCode": 200,
            "body": json.dumps({
                "gdm_type": gdm_type
            }),
            "headers": {"Content-Type": "application/json"}
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
