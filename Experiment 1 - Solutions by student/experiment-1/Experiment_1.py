import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------
# 1. Create output folder
# ---------------------------------------------------------

output_folder = "output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# ---------------------------------------------------------
# 2. Load the image
# ---------------------------------------------------------

image_path = "/home/pawan/Desktop/Computer-Vision-Lab-2026-27-ODD-Sem-3rd-Year-Section-C/Experiment 1 - Solutions by student/experiment-1/computervision.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    print("Please place input_image.jpg inside the input folder.")
    exit()

print("Image loaded successfully.")


# ---------------------------------------------------------
# 3. Display image using OpenCV
# ---------------------------------------------------------

cv2.imshow("Original Image - OpenCV", image)

cv2.waitKey(1000)
cv2.destroyAllWindows()


# ---------------------------------------------------------
# 4. Display image using Matplotlib
# ---------------------------------------------------------

# OpenCV loads images in BGR format.
# Matplotlib expects RGB format.

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(7, 5))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()


# ---------------------------------------------------------
# 5. Examine image properties
# ---------------------------------------------------------

height, width, channels = image.shape

print("\nIMAGE PROPERTIES")
print("-------------------------")
print("Width       :", width)
print("Height      :", height)
print("Channels    :", channels)
print("Resolution  :", width, "x", height)
print("Data Type   :", image.dtype)
print("Total Pixels:", width * height)


# ---------------------------------------------------------
# 6. Save image in JPEG and PNG
# ---------------------------------------------------------

cv2.imwrite("output/saved_image.jpg", image)
cv2.imwrite("output/saved_image.png", image)

print("\nImage saved in JPEG and PNG formats.")


# ---------------------------------------------------------
# 7. Convert image to Grayscale
# ---------------------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite("output/grayscale.jpg", gray)

plt.figure(figsize=(7, 5))
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()


# ---------------------------------------------------------
# 8. Convert image to HSV
# ---------------------------------------------------------

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imwrite("output/hsv.jpg", hsv)

# Convert HSV back to RGB only for proper display
hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

plt.figure(figsize=(7, 5))
plt.imshow(hsv_rgb)
plt.title("HSV Image")
plt.axis("off")
plt.show()


# ---------------------------------------------------------
# 9. Convert image to LAB
# ---------------------------------------------------------

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

cv2.imwrite("output/lab.jpg", lab)

# Convert LAB back to RGB for display
lab_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

plt.figure(figsize=(7, 5))
plt.imshow(lab_rgb)
plt.title("LAB Image")
plt.axis("off")
plt.show()


# ---------------------------------------------------------
# 10. Resize image
# ---------------------------------------------------------

resized = cv2.resize(image, (400, 300))

cv2.imwrite("output/resized.jpg", resized)

plt.figure(figsize=(6, 4))
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.title("Resized Image")
plt.axis("off")
plt.show()


# ---------------------------------------------------------
# 11. Rotate image
# ---------------------------------------------------------

height, width = image.shape[:2]

center = (width // 2, height // 2)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    45,
    1.0
)

rotated = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height)
)

cv2.imwrite("output/rotated.jpg", rotated)

plt.figure(figsize=(6, 5))
plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
plt.title("Rotated Image - 45 Degrees")
plt.axis("off")
plt.show()


# ---------------------------------------------------------
# 12. Horizontal Flip
# ---------------------------------------------------------

horizontal_flip = cv2.flip(image, 1)

cv2.imwrite(
    "output/horizontal_flip.jpg",
    horizontal_flip
)

plt.figure(figsize=(6, 5))
plt.imshow(cv2.cvtColor(horizontal_flip, cv2.COLOR_BGR2RGB))
plt.title("Horizontal Flip")
plt.axis("off")
plt.show()


# ---------------------------------------------------------
# 13. Vertical Flip
# ---------------------------------------------------------

vertical_flip = cv2.flip(image, 0)

cv2.imwrite(
    "output/vertical_flip.jpg",
    vertical_flip
)

plt.figure(figsize=(6, 5))
plt.imshow(cv2.cvtColor(vertical_flip, cv2.COLOR_BGR2RGB))
plt.title("Vertical Flip")
plt.axis("off")
plt.show()


# ---------------------------------------------------------
# 14. Image Complement / Negative
# ---------------------------------------------------------

negative = 255 - image

cv2.imwrite(
    "output/negative.jpg",
    negative
)

plt.figure(figsize=(6, 5))
plt.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB))
plt.title("Negative Image")
plt.axis("off")
plt.show()


# ---------------------------------------------------------
# 15. Crop Region of Interest (ROI)
# ---------------------------------------------------------

# ROI coordinates
# Change these values according to your image.

x1 = 50
y1 = 50
x2 = min(width, 350)
y2 = min(height, 300)

roi = image[y1:y2, x1:x2]

cv2.imwrite(
    "output/roi.jpg",
    roi
)

plt.figure(figsize=(6, 5))
plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
plt.title("Region of Interest (ROI)")
plt.axis("off")
plt.show()


# ---------------------------------------------------------
# 16. Display comparison of original and processed images
# ---------------------------------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.title("Resized")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
plt.title("Rotated")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(horizontal_flip, cv2.COLOR_BGR2RGB))
plt.title("Horizontal Flip")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB))
plt.title("Negative")
plt.axis("off")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 17. Final message
# ---------------------------------------------------------

print("\n======================================")
print("Experiment 1 completed successfully!")
print("======================================")

print("\nGenerated files:")

for file in os.listdir(output_folder):
    print("-", file)