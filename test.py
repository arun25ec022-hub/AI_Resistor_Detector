from ultralytics import YOLO

# Load our trained model
model = YOLO(
    "runs/detect/models/resistor_detector_fast-3/weights/best.pt"
)

# Test an image
results = model.predict(
    source="test_resistor.jpg",
    conf=0.25,
    save=True
)

# Print predictions
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(f"Detected: {class_name}")
        print(f"Confidence: {confidence:.2%}")