from flask import Flask, request, jsonify
import requests
import base64
import io
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim


app = Flask(__name__)
MODEL_SERVICE_URL = "http://localhost:5001/predict"
SSIM_THRESHOLD = 0.8

def compare_structural_similarity(image1_path: str, image2_path: str) -> float:
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    image1 = cv2.resize(image1, (300, 300))
    image2 = cv2.resize(image2, (300, 300))
    score, _ = ssim(image1, image2, full= True)
    return score

@app.route("/detect", methods=["POST"])
def detect():
    try: 
        data = request.get_json()  
        image1_path = data.get("image1")
        image2_path = data.get("image2")
        score = compare_structural_similarity(image1_path, image2_path)

        if score > SSIM_THRESHOLD:
            return jsonify({"message": "No major changes"})
        
        response = requests.post(MODEL_SERVICE_URL, json={"image_path": image2_path})
        return response.json()
    
    except Exception as e:
        return jsonify({"error": str(e)}, 500)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug= True)