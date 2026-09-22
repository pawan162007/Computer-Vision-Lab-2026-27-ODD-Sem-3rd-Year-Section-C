import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time


# =========================================================
# 1. Create output folder
# =========================================================

os.makedirs("output", exist_ok=True)


# =========================================================
# 2. Load Image
# =========================================================

image_path = "input/input_image.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    print("Please put input_image.jpg inside the input folder.")
    exit()

print("Image loaded successfully.")


# =========================================================
# 3. Convert to Grayscale
# =========================================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite(
    "output/original_grayscale.jpg",
    gray
)


# =========================================================
# 4. Display Original Image
# =========================================================

plt.figure(figsize=(7, 5))
plt.imshow(gray, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")

plt.savefig(
    "output/original_image.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 5. Add Salt and Pepper Noise
# =========================================================

noisy_image = gray.copy()

noise_amount = 0.03

num_pixels = int(noise_amount * gray.size)

# Salt noise
salt_coordinates = (
    np.random.randint(0, gray.shape[0], num_pixels),
    np.random.randint(0, gray.shape[1], num_pixels)
)

noisy_image[salt_coordinates] = 255


# Pepper noise
pepper_coordinates = (
    np.random.randint(0, gray.shape[0], num_pixels),
    np.random.randint(0, gray.shape[1], num_pixels)
)

noisy_image[pepper_coordinates] = 0

cv2.imwrite(
    "output/salt_pepper_noisy.jpg",
    noisy_image
)


# =========================================================
# 6. Gaussian Filter
# =========================================================

start_time = time.perf_counter()

gaussian = cv2.GaussianBlur(
    noisy_image,
    (5, 5),
    0
)

gaussian_time = time.perf_counter() - start_time

cv2.imwrite(
    "output/gaussian_blur.jpg",
    gaussian
)


# =========================================================
# 7. Median Filter
# =========================================================

start_time = time.perf_counter()

median = cv2.medianBlur(
    noisy_image,
    5
)

median_time = time.perf_counter() - start_time

cv2.imwrite(
    "output/median_filter.jpg",
    median
)


# =========================================================
# 8. Average / Mean Filter
# =========================================================

start_time = time.perf_counter()

average = cv2.blur(
    noisy_image,
    (5, 5)
)

average_time = time.perf_counter() - start_time

cv2.imwrite(
    "output/average_filter.jpg",
    average
)


# =========================================================
# 9. Laplacian Filter
# =========================================================

laplacian = cv2.Laplacian(
    gray,
    cv2.CV_64F
)

laplacian = cv2.convertScaleAbs(
    laplacian
)

cv2.imwrite(
    "output/laplacian.jpg",
    laplacian
)


# =========================================================
# 10. Sobel X
# =========================================================

sobel_x = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

sobel_x = cv2.convertScaleAbs(
    sobel_x
)

cv2.imwrite(
    "output/sobel_x.jpg",
    sobel_x
)


# =========================================================
# 11. Sobel Y
# =========================================================

sobel_y = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

sobel_y = cv2.convertScaleAbs(
    sobel_y
)

cv2.imwrite(
    "output/sobel_y.jpg",
    sobel_y
)


# =========================================================
# 12. Combined Sobel
# =========================================================

sobel_combined = cv2.addWeighted(
    sobel_x,
    0.5,
    sobel_y,
    0.5,
    0
)

cv2.imwrite(
    "output/sobel_combined.jpg",
    sobel_combined
)


# =========================================================
# 13. Low-Pass Filter Comparison
# =========================================================

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(noisy_image, cmap="gray")
plt.title("Salt and Pepper Noise")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gaussian, cmap="gray")
plt.title("Gaussian Filter")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(median, cmap="gray")
plt.title("Median Filter")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(average, cmap="gray")
plt.title("Average Filter")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "output/low_pass_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 14. High-Pass Filter Comparison
# =========================================================

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(sobel_x, cmap="gray")
plt.title("Sobel X")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(sobel_y, cmap="gray")
plt.title("Sobel Y")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "output/high_pass_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 15. Complete Comparison
# =========================================================

plt.figure(figsize=(16, 10))

images = [
    (gray, "Original"),
    (noisy_image, "Salt and Pepper Noise"),
    (gaussian, "Gaussian Filter"),
    (median, "Median Filter"),
    (average, "Average Filter"),
    (laplacian, "Laplacian"),
    (sobel_x, "Sobel X"),
    (sobel_y, "Sobel Y")
]

for i, (img, title) in enumerate(images):

    plt.subplot(2, 4, i + 1)

    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "output/final_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 16. Filtering Time Comparison
# =========================================================

print("\nFILTERING TIME COMPARISON")
print("--------------------------------")

print(
    f"Gaussian Filter : {gaussian_time:.6f} seconds"
)

print(
    f"Median Filter   : {median_time:.6f} seconds"
)

print(
    f"Average Filter  : {average_time:.6f} seconds"
)


# =========================================================
# 17. Image Information
# =========================================================

height, width = gray.shape

print("\nIMAGE INFORMATION")
print("--------------------------------")

print("Width      :", width)
print("Height     :", height)
print("Resolution :", width, "x", height)
print("Data Type  :", gray.dtype)


# =========================================================
# 18. Final Message
# =========================================================

print("\n======================================")
print("Experiment 3 completed successfully!")
print("======================================")

print("\nGenerated files:")

for file in sorted(os.listdir("output")):
    print("-", file)