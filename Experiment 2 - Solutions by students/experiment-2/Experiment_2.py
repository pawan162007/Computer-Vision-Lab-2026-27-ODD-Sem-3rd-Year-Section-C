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
# 2. Load image
# ---------------------------------------------------------

image_path = "input_image.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    print("Please place input_image.jpg inside the input folder.")
    exit()

print("Image loaded successfully.")


# ---------------------------------------------------------
# 3. Convert image to grayscale
# ---------------------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite("output/original_grayscale.jpg", gray)


# ---------------------------------------------------------
# 4. Display original grayscale image
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.imshow(gray, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")

plt.show()


# ---------------------------------------------------------
# 5. Generate Original Histogram
# ---------------------------------------------------------

hist_original = cv2.calcHist(
    [gray],
    [0],
    None,
    [256],
    [0, 256]
)

plt.figure(figsize=(8, 5))

plt.plot(hist_original)

plt.title("Histogram of Original Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.grid()

plt.savefig(
    "output/original_histogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 6. Contrast Stretching
# ---------------------------------------------------------

minimum = np.min(gray)
maximum = np.max(gray)

contrast_stretched = cv2.normalize(
    gray,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

cv2.imwrite(
    "output/contrast_stretched.jpg",
    contrast_stretched
)

print("\nContrast Stretching")
print("-------------------------")
print("Minimum intensity:", minimum)
print("Maximum intensity:", maximum)


# ---------------------------------------------------------
# 7. Histogram Equalization
# ---------------------------------------------------------

hist_equalized = cv2.equalizeHist(gray)

cv2.imwrite(
    "output/histogram_equalized.jpg",
    hist_equalized
)


# ---------------------------------------------------------
# 8. CLAHE
# ---------------------------------------------------------

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_image = clahe.apply(gray)

cv2.imwrite(
    "output/clahe.jpg",
    clahe_image
)


# ---------------------------------------------------------
# 9. Histograms of processed images
# ---------------------------------------------------------

hist_stretched = cv2.calcHist(
    [contrast_stretched],
    [0],
    None,
    [256],
    [0, 256]
)

hist_equalized_data = cv2.calcHist(
    [hist_equalized],
    [0],
    None,
    [256],
    [0, 256]
)

hist_clahe = cv2.calcHist(
    [clahe_image],
    [0],
    None,
    [256],
    [0, 256]
)


# ---------------------------------------------------------
# 10. Compare histograms
# ---------------------------------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(hist_original)
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(hist_stretched)
plt.title("Contrast Stretching Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(hist_equalized_data)
plt.title("Histogram Equalization")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(hist_clahe)
plt.title("CLAHE Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.grid()

plt.tight_layout()

plt.savefig(
    "output/histogram_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 11. Compare all enhanced images
# ---------------------------------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(contrast_stretched, cmap="gray")
plt.title("Contrast Stretching")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(hist_equalized, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(clahe_image, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "output/image_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 12. Individual comparison
# ---------------------------------------------------------

plt.figure(figsize=(14, 5))

plt.subplot(1, 4, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(contrast_stretched, cmap="gray")
plt.title("Contrast Stretching")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(hist_equalized, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(clahe_image, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "output/final_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ---------------------------------------------------------
# 13. Image information
# ---------------------------------------------------------

height, width = gray.shape

print("\nIMAGE INFORMATION")
print("-------------------------")
print("Width       :", width)
print("Height      :", height)
print("Resolution  :", width, "x", height)
print("Data Type   :", gray.dtype)


# ---------------------------------------------------------
# 14. Final message
# ---------------------------------------------------------

print("\n======================================")
print("Experiment 2 completed successfully!")
print("======================================")

print("\nGenerated files:")

for file in os.listdir(output_folder):
    print("-", file)