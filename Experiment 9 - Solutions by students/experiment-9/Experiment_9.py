from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import time

# Create output folder
os.makedirs("output", exist_ok=True)

# Input image
image_path = "input/input_image.jpg"

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

print("Image loaded successfully.")
print("Image Size:", image.shape)

# Load YOLOv8 model
print("\nLoading YOLOv8 model...")
model = YOLO("yolov8n.pt")
print("YOLOv8 model loaded successfully.")

# Run detection
print("\nRunning object detection...")

start_time = time.time()

results = model.predict(
    source=image_path,
    conf=0.25,
    save=False,
    verbose=False
)

inference_time = time.time() - start_time

result = results[0]

# Draw detected objects
detected_image = result.plot()

# Convert BGR to RGB
detected_image_rgb = cv2.cvtColor(
    detected_image,
    cv2.COLOR_BGR2RGB
)

# Save detected image
cv2.imwrite(
    "output/detected_objects.jpg",
    detected_image
)

# Display detection result
plt.figure(figsize=(12, 8))
plt.imshow(detected_image_rgb)
plt.title("YOLOv8 Object Detection")
plt.axis("off")
plt.savefig(
    "output/object_detection_result.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# Detection information
boxes = result.boxes

print("\n========================================")
print("DETECTION RESULTS")
print("========================================")

if boxes is None or len(boxes) == 0:

    print("No objects detected.")

else:

    print("Total Objects Detected:", len(boxes))

    print("\nDetected Objects:")

    for i, box in enumerate(boxes):

        class_id = int(box.cls[0].item())

        confidence = float(
            box.conf[0].item()
        )

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_name = model.names[class_id]

        print(
            f"{i + 1}. {class_name} | "
            f"Confidence: {confidence:.2f} | "
            f"Box: ({int(x1)}, {int(y1)}) - "
            f"({int(x2)}, {int(y2)})"
        )

# Object summary
object_names = []

if boxes is not None:

    for box in boxes:

        class_id = int(box.cls[0].item())

        object_names.append(
            model.names[class_id]
        )

print("\n========================================")
print("OBJECT SUMMARY")
print("========================================")

if object_names:

    unique_objects = sorted(
        set(object_names)
    )

    for name in unique_objects:

        count = object_names.count(name)

        print(
            f"{name}: {count}"
        )

else:

    print("No objects detected.")

# Original vs detected comparison
original_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(detected_image_rgb)
plt.title("YOLOv8 Detection")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "output/original_vs_detection.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Performance
print("\n========================================")
print("PERFORMANCE")
print("========================================")

print(
    "Inference Time:",
    round(inference_time, 4),
    "seconds"
)

if inference_time > 0:

    fps = 1 / inference_time

    print(
        "Approx. Processing Rate:",
        round(fps, 2),
        "images/second"
    )

print("\n========================================")
print("Experiment 9 completed successfully!")
print("========================================")

print("\nGenerated files:")

for file in sorted(os.listdir("output")):

    print("-", file)