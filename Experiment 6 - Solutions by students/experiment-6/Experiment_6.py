import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from sklearn.cluster import KMeans


# =========================================================
# 1. Create Output Folder
# =========================================================

os.makedirs("output", exist_ok=True)


# =========================================================
# 2. Load Image
# =========================================================

image_path = "input/input_image.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    print("Please put input_image.jpg inside input folder.")
    exit()

print("Image loaded successfully.")

# Convert BGR to RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# =========================================================
# 3. Convert to Grayscale
# =========================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

print("Image Size:", image.shape)


# =========================================================
# 4. Gaussian Blur
# =========================================================

blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

cv2.imwrite(
    "output/grayscale.jpg",
    gray
)

cv2.imwrite(
    "output/blurred.jpg",
    blurred
)


# =========================================================
# 5. Global Thresholding
# =========================================================

start_time = time.time()

_, global_threshold = cv2.threshold(
    blurred,
    127,
    255,
    cv2.THRESH_BINARY
)

global_time = time.time() - start_time

cv2.imwrite(
    "output/global_threshold.jpg",
    global_threshold
)

print("\nGlobal Thresholding Time:",
      round(global_time, 4), "seconds")


# =========================================================
# 6. Otsu's Thresholding
# =========================================================

start_time = time.time()

otsu_value, otsu_threshold = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

otsu_time = time.time() - start_time

cv2.imwrite(
    "output/otsu_threshold.jpg",
    otsu_threshold
)

print("Otsu Threshold Value:", otsu_value)
print("Otsu Processing Time:",
      round(otsu_time, 4), "seconds")


# =========================================================
# 7. Adaptive Thresholding
# =========================================================

start_time = time.time()

adaptive_threshold = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

adaptive_time = time.time() - start_time

cv2.imwrite(
    "output/adaptive_threshold.jpg",
    adaptive_threshold
)

print("Adaptive Thresholding Time:",
      round(adaptive_time, 4), "seconds")


# =========================================================
# 8. Watershed Segmentation
# =========================================================

print("\nRunning Watershed Segmentation...")

start_time = time.time()

# Convert threshold result to binary
binary = otsu_threshold.copy()

# Remove noise
kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)

# Find sure background
sure_background = cv2.dilate(
    opening,
    kernel,
    iterations=3
)

# Distance transform
distance = cv2.distanceTransform(
    opening,
    cv2.DIST_L2,
    5
)

# Find sure foreground
_, sure_foreground = cv2.threshold(
    distance,
    0.5 * distance.max(),
    255,
    0
)

sure_foreground = np.uint8(sure_foreground)

# Unknown region
unknown = cv2.subtract(
    sure_background,
    sure_foreground
)

# Marker labelling
num_labels, markers = cv2.connectedComponents(
    sure_foreground
)

markers = markers + 1

markers[unknown == 255] = 0

# Apply Watershed
watershed_image = image.copy()

markers = cv2.watershed(
    watershed_image,
    markers
)

# Mark boundaries in red
watershed_result = image_rgb.copy()

watershed_result[markers == -1] = [255, 0, 0]

watershed_time = time.time() - start_time

cv2.imwrite(
    "output/watershed.jpg",
    cv2.cvtColor(
        watershed_result,
        cv2.COLOR_RGB2BGR
    )
)

print("Watershed Processing Time:",
      round(watershed_time, 4), "seconds")


# =========================================================
# 9. K-Means Clustering
# =========================================================

print("\nRunning K-Means Clustering...")

start_time = time.time()

# Resize image for faster processing
small_image = cv2.resize(
    image_rgb,
    (300, 300)
)

# Convert pixels into a list
pixel_data = small_image.reshape(
    (-1, 3)
)

pixel_data = np.float32(pixel_data)

# Number of clusters
k = 4

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(
    pixel_data
)

centers = np.uint8(
    kmeans.cluster_centers_
)

segmented_data = centers[labels]

segmented_image = segmented_data.reshape(
    small_image.shape
)

kmeans_time = time.time() - start_time

cv2.imwrite(
    "output/kmeans_segmentation.jpg",
    cv2.cvtColor(
        segmented_image,
        cv2.COLOR_RGB2BGR
    )
)

print("Number of Clusters:", k)

print("K-Means Processing Time:",
      round(kmeans_time, 4), "seconds")


# =========================================================
# 10. Histogram Comparison
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(
    gray.ravel(),
    bins=256,
    range=[0, 256]
)

plt.title("Grayscale Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.savefig(
    "output/grayscale_histogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 11. Thresholding Comparison
# =========================================================

plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Grayscale")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(global_threshold, cmap="gray")
plt.title("Global Thresholding")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(otsu_threshold, cmap="gray")
plt.title("Otsu's Thresholding")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(adaptive_threshold, cmap="gray")
plt.title("Adaptive Thresholding")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "output/threshold_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 12. Watershed Visualization
# =========================================================

plt.figure(figsize=(10, 6))

plt.imshow(watershed_result)

plt.title("Watershed Segmentation")
plt.axis("off")

plt.savefig(
    "output/watershed_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 13. K-Means Visualization
# =========================================================

plt.figure(figsize=(10, 6))

plt.imshow(segmented_image)

plt.title("K-Means Color Segmentation")

plt.axis("off")

plt.savefig(
    "output/kmeans_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 14. Complete Comparison
# =========================================================

plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(global_threshold, cmap="gray")
plt.title("Global Threshold")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(otsu_threshold, cmap="gray")
plt.title("Otsu Threshold")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(adaptive_threshold, cmap="gray")
plt.title("Adaptive Threshold")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(watershed_result)
plt.title("Watershed")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(segmented_image)
plt.title("K-Means")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "output/final_segmentation_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 15. Final Results
# =========================================================

print("\n============================================")
print("IMAGE SEGMENTATION RESULTS")
print("============================================")

print("Global Threshold Time    :",
      round(global_time, 4), "seconds")

print("Otsu Threshold Time      :",
      round(otsu_time, 4), "seconds")

print("Adaptive Threshold Time  :",
      round(adaptive_time, 4), "seconds")

print("Watershed Time            :",
      round(watershed_time, 4), "seconds")

print("K-Means Time              :",
      round(kmeans_time, 4), "seconds")

print("\n============================================")
print("Experiment 6 completed successfully!")
print("============================================")


# =========================================================
# 16. Generated Files
# =========================================================

print("\nGenerated files:")

for file in sorted(os.listdir("output")):
    print("-", file)