from object_detection.mobilenet_v2_ssd import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
import cv2
import time
import os

# Set up relative paths
current_dir = os.path.dirname(os.path.abspath(__file__))  
label_path = os.path.join(current_dir, "models_parameters", "labels.txt")
model_path = os.path.join(current_dir, "models_parameters", "back_model.pth")
input_image_path = os.path.join(current_dir, "image.jpg")
output_image_path = os.path.join(current_dir, "output_with_time.jpg")

class_names = [name.strip() for name in open(label_path).readlines()]
net = create_mobilenetv2_ssd_lite(2, is_test=True)
net.load(model_path, load_quantized= False)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
img = cv2.imread(input_image_path)
# img = cv2.flip(img, -1)
# img = img[650:1200, 100:1400]
boxes, labels, probs = predictor.predict(img, 10, 0.2)
for i in range(boxes.size(0)):
    box = boxes[i, :].cpu().numpy()  # Ensure box is in a NumPy array
    x1, y1, x2, y2 = map(int, box)   # Convert to integers
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 4)
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(img, label,
                (x1 + 20, y1 + 40),  # Text position
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),  # color
                2)  # line type
    

# Save the image with bounding boxes and labels
cv2.imwrite(output_image_path, img)

















# for i in range(boxes.size(0)):
#     box = boxes[i, :].cpu().numpy()
#     x1, y1, x2, y2 = map(int, box)
#     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 4)
#     label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
#     cv2.putText(img, label,
#                 (x1 + 20, y1 + 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (255, 0, 255),
#                 2)

# inference_time = inference_time + 1
# time_label = f"Inference Time: {inference_time:.3f}s"
# cv2.putText(img, time_label,
#             (10, 50),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1.5,
#             (0, 255, 0),
#             3)

# cv2.imwrite(output_image_path, img)