import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
model = load_model("best_model.h5")
print("✅ Model loaded")

# Define the class names (must match training order)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # Update if different

# Confidence threshold (only show label if prediction confidence ≥ 90%)
CONFIDENCE_THRESHOLD = 0.80

# Classify a single image (ROI from the frame)
def classify_frame(frame):
    # Resize to model input size
    resized = cv2.resize(frame, (160, 160))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    processed = preprocess_input(rgb.astype(np.float32))
    input_tensor = np.expand_dims(processed, axis=0)

    # Predict using the model
    preds = model.predict(input_tensor, verbose=0)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id]

    if confidence >= CONFIDENCE_THRESHOLD:
        label = class_names[class_id]
        return label, confidence
    else:
        return None, confidence

# Start the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🚀 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame. Exiting...")
        break

    # Optional: Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Define a fixed region of interest (ROI) box
    x, y, w, h = 100, 100, 300, 300
    roi = frame[y:y+h, x:x+w]

    # Classify the ROI
    label, confidence = classify_frame(roi)

    # Draw bounding box and display label/confidence
    color = (0, 255, 0) if label else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    if label:
        text = f"{label} ({confidence:.2%})"
    else:
        text = "Unknown"

    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show the frame
    cv2.imshow("Smart Waste Classifier", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam stopped")
