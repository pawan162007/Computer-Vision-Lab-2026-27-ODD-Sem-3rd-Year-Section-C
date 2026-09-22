import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from skimage.feature import hog
from skimage import exposure


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


# =========================================================
# 3. Convert Image to RGB and Grayscale
# =========================================================

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Image Size:", image.shape)
print("Grayscale conversion completed.")


# =========================================================
# 4. Save Grayscale Image
# =========================================================

cv2.imwrite(
    "output/grayscale.jpg",
    gray
)


# =========================================================
# 5. SIFT Feature Extraction
# =========================================================

print("\nRunning SIFT...")

start_time = time.time()

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(
    gray,
    None
)

sift_time = time.time() - start_time

print("SIFT Keypoints:", len(keypoints))
print("SIFT Descriptor Shape:", descriptors.shape if descriptors is not None else None)
print("SIFT Processing Time:", round(sift_time, 4), "seconds")


# =========================================================
# 6. Draw SIFT Keypoints
# =========================================================

sift_image = cv2.drawKeypoints(
    image_rgb,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(10, 6))
plt.imshow(sift_image)
plt.title("SIFT Keypoints")
plt.axis("off")

plt.savefig(
    "output/sift_keypoints.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 7. HOG Feature Extraction
# =========================================================

print("\nRunning HOG...")

start_time = time.time()

gray_float = gray / 255.0

hog_features, hog_image = hog(
    gray_float,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    visualize=True
)

hog_time = time.time() - start_time

print("HOG Feature Length:", len(hog_features))
print("HOG Processing Time:", round(hog_time, 4), "seconds")


# =========================================================
# 8. Improve HOG Visualization
# =========================================================

hog_image_rescaled = exposure.rescale_intensity(
    hog_image,
    in_range=(0, 10)
)


plt.figure(figsize=(10, 6))
plt.imshow(hog_image_rescaled, cmap="gray")
plt.title("HOG Visualization")
plt.axis("off")

plt.savefig(
    "output/hog_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 9. Original vs SIFT vs HOG
# =========================================================

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")


plt.subplot(1, 3, 2)
plt.imshow(sift_image)
plt.title("SIFT Keypoints")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(hog_image_rescaled, cmap="gray")
plt.title("HOG Features")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    "output/feature_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 10. Create Second Similar Image
# =========================================================

# Slightly resize and rotate the original image
height, width = gray.shape

center = (width // 2, height // 2)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    5,
    1.0
)

second_image = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height)
)

cv2.imwrite(
    "output/second_similar_image.jpg",
    second_image
)


# =========================================================
# 11. SIFT Features for Second Image
# =========================================================

second_gray = cv2.cvtColor(
    second_image,
    cv2.COLOR_BGR2GRAY
)

keypoints2, descriptors2 = sift.detectAndCompute(
    second_gray,
    None
)

print("\nSecond Image SIFT Keypoints:", len(keypoints2))


# =========================================================
# 12. SIFT Image Matching
# =========================================================

print("\nPerforming SIFT Image Matching...")

if descriptors is not None and descriptors2 is not None:

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=True
    )

    matches = matcher.match(
        descriptors,
        descriptors2
    )

    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    # Select best 50 matches
    good_matches = matches[:50]

    match_image = cv2.drawMatches(
        image_rgb,
        keypoints,
        cv2.cvtColor(second_image, cv2.COLOR_BGR2RGB),
        keypoints2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(16, 8))
    plt.imshow(match_image)
    plt.title("SIFT Feature Matching")
    plt.axis("off")

    plt.savefig(
        "output/sift_matching.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Total Matches:", len(matches))
    print("Best Matches Displayed:", len(good_matches))

else:
    print("Descriptors could not be calculated.")


# =========================================================
# 13. Feature Statistics
# =========================================================

print("\n========================================")
print("FEATURE EXTRACTION RESULTS")
print("========================================")

print("SIFT Keypoints       :", len(keypoints))
print("SIFT Descriptor Size :", descriptors.shape if descriptors is not None else None)

print("HOG Feature Length   :", len(hog_features))

print("SIFT Time             :", round(sift_time, 4), "seconds")
print("HOG Time              :", round(hog_time, 4), "seconds")


# =========================================================
# 14. Final Comparison Plot
# =========================================================

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")


plt.subplot(2, 2, 2)
plt.imshow(sift_image)
plt.title("SIFT Keypoints")
plt.axis("off")


plt.subplot(2, 2, 3)
plt.imshow(hog_image_rescaled, cmap="gray")
plt.title("HOG Descriptor")
plt.axis("off")


plt.subplot(2, 2, 4)
plt.imshow(
    cv2.cvtColor(second_image, cv2.COLOR_BGR2RGB)
)
plt.title("Similar / Rotated Image")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    "output/final_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 15. Final Message
# =========================================================

print("\n========================================")
print("Experiment 5 completed successfully!")
print("========================================")

print("\nGenerated files:")

for file in sorted(os.listdir("output")):
    print("-", file)