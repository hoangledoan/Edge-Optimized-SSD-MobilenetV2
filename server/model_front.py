import tvm
import numpy as np
import cv2
import torch
import time
from training.data_preprocessing import PredictionTransform
from utils import box_utils
from flask import Flask, request, jsonify

app = Flask(__name__)

COMPILED_MODEL_PATH = "tvm_model/front_compiled_model.so"
def run_inference(compiled_model_path, img, prob_threshold=0.3):
    """
    Run inference using compiled TVM model and return detected bounding boxes.
    """
    # Load compiled model
    lib = tvm.runtime.load_module(compiled_model_path)
    dev = tvm.cpu(0)
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    # Preprocess image
    transform = PredictionTransform(
        size=300,
        mean=np.array([127, 127, 127]),
        std=128.0
    )


    height, width, _ = img.shape
    image = transform(img)
    images = image.unsqueeze(0)

    # Run inference
    module.set_input("input0", tvm.nd.array(images.numpy()))
    module.run()
    scores = torch.from_numpy(module.get_output(0).numpy())[0]
    boxes = torch.from_numpy(module.get_output(1).numpy())[0]
    # Post-process
    picked_box_probs = []
    for class_index in range(1, scores.size(1)):
        probs = scores[:, class_index]
        mask = probs > prob_threshold
        if probs[mask].size(0) == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = torch.cat([subset_boxes, probs[mask].reshape(-1, 1)], dim=1)
        box_probs = box_utils.nms(box_probs, None, 
                                  score_threshold=prob_threshold, 
                                  iou_threshold=0.45, 
                                  sigma=0.5, 
                                  top_k=10, 
                                  candidate_size=200)
        picked_box_probs.append(box_probs)

    if not picked_box_probs:
        return torch.tensor([])  

    picked_box_probs = torch.cat(picked_box_probs)

    picked_box_probs[:, [0, 2]] *= width
    picked_box_probs[:, [1, 3]] *= height
    return picked_box_probs[:, :4]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "image_path" not in data:
            return jsonify({"error": "No image path provided"}), 400

        image_path = data["image_path"]
        img = cv2.imread(image_path)
        img = cv2.flip(img, -1)
        img = img[650:1200, 100:1400]
        boxes = run_inference(COMPILED_MODEL_PATH, img)

        for i in range(boxes.size(0)):
            box = boxes[i, :].cpu().numpy() 
            x1, y1, x2, y2 = map(int, box)   
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 4)
            label = f"people"
            cv2.putText(img, label,
                        (x1 + 20, y1 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  
                        (255, 0, 255),  
                        2)  
            

        # Save the image with bounding boxes and labels
        cv2.imwrite(f"/app/{image_path}_boxes.jpg", img)
        
        return jsonify({
            "number_of_people": len(boxes)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True) 
