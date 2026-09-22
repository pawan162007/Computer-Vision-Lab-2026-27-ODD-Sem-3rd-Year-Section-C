import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# =========================================================
# 1. Create output folder
# =========================================================

os.makedirs("output", exist_ok=True)


# =========================================================
# 2. Load Image
# =========================================================

image_path = "input/input_image.jpg"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found!")
    print("Please put input_image.jpg inside input folder.")
    exit()

print("Image loaded successfully.")


# =========================================================
# 3. Display Original Image
# =========================================================

plt.figure(figsize=(7, 5))

plt.imshow(image, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")

plt.savefig(
    "output/original_image.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 4. Compute DFT
# =========================================================

dft = cv2.dft(
    np.float32(image),
    flags=cv2.DFT_COMPLEX_OUTPUT
)


# =========================================================
# 5. Shift Zero Frequency to Center
# =========================================================

dft_shift = np.fft.fftshift(dft)


# =========================================================
# 6. Calculate Magnitude Spectrum
# =========================================================

magnitude_spectrum = cv2.magnitude(
    dft_shift[:, :, 0],
    dft_shift[:, :, 1]
)

magnitude_spectrum = 20 * np.log(
    magnitude_spectrum + 1
)


# Save magnitude spectrum

plt.figure(figsize=(8, 6))

plt.imshow(
    magnitude_spectrum,
    cmap="gray"
)

plt.title("Magnitude Spectrum")

plt.axis("off")

plt.savefig(
    "output/magnitude_spectrum.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 7. Image Dimensions
# =========================================================

rows, cols = image.shape

crow = rows // 2
ccol = cols // 2

print("\nIMAGE INFORMATION")
print("--------------------------------")
print("Width      :", cols)
print("Height     :", rows)
print("Resolution :", cols, "x", rows)


# =========================================================
# 8. Low-Pass Frequency Filter
# =========================================================

# Create mask with zeros
low_pass_mask = np.zeros(
    (rows, cols, 2),
    np.uint8
)

# Radius of low-pass filter
radius = 40

cv2.circle(
    low_pass_mask,
    (ccol, crow),
    radius,
    (1, 1),
    -1
)


# Apply Low-Pass Filter

low_pass_dft = dft_shift * low_pass_mask


# =========================================================
# 9. Inverse Fourier Transform for Low-Pass
# =========================================================

low_pass_shift = np.fft.ifftshift(
    low_pass_dft
)

low_pass_image = cv2.idft(
    low_pass_shift
)

low_pass_image = cv2.magnitude(
    low_pass_image[:, :, 0],
    low_pass_image[:, :, 1]
)

# Normalize to 0-255

low_pass_image = cv2.normalize(
    low_pass_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

low_pass_image = np.uint8(
    low_pass_image
)

cv2.imwrite(
    "output/low_pass_filtered.jpg",
    low_pass_image
)


# =========================================================
# 10. High-Pass Frequency Filter
# =========================================================

high_pass_mask = np.ones(
    (rows, cols, 2),
    np.uint8
)

# Remove low-frequency center

cv2.circle(
    high_pass_mask,
    (ccol, crow),
    radius,
    (0, 0),
    -1
)


# Apply High-Pass Filter

high_pass_dft = dft_shift * high_pass_mask


# =========================================================
# 11. Inverse Fourier Transform for High-Pass
# =========================================================

high_pass_shift = np.fft.ifftshift(
    high_pass_dft
)

high_pass_image = cv2.idft(
    high_pass_shift
)

high_pass_image = cv2.magnitude(
    high_pass_image[:, :, 0],
    high_pass_image[:, :, 1]
)

# Normalize

high_pass_image = cv2.normalize(
    high_pass_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

high_pass_image = np.uint8(
    high_pass_image
)

cv2.imwrite(
    "output/high_pass_filtered.jpg",
    high_pass_image
)


# =========================================================
# 12. Display Low-Pass and High-Pass Results
# =========================================================

plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)

plt.imshow(
    image,
    cmap="gray"
)

plt.title("Original")
plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    low_pass_image,
    cmap="gray"
)

plt.title("Low-Pass Filtered")
plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    high_pass_image,
    cmap="gray"
)

plt.title("High-Pass Filtered")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    "output/filter_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 13. Frequency Masks Visualization
# =========================================================

low_mask_display = low_pass_mask[:, :, 0] * 255
high_mask_display = high_pass_mask[:, :, 0] * 255


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.imshow(
    low_mask_display,
    cmap="gray"
)

plt.title("Low-Pass Frequency Mask")
plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    high_mask_display,
    cmap="gray"
)

plt.title("High-Pass Frequency Mask")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    "output/frequency_masks.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 14. Complete Comparison
# =========================================================

plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)

plt.imshow(
    image,
    cmap="gray"
)

plt.title("Original")
plt.axis("off")


plt.subplot(2, 3, 2)

plt.imshow(
    magnitude_spectrum,
    cmap="gray"
)

plt.title("Magnitude Spectrum")
plt.axis("off")


plt.subplot(2, 3, 3)

plt.imshow(
    low_mask_display,
    cmap="gray"
)

plt.title("Low-Pass Mask")
plt.axis("off")


plt.subplot(2, 3, 4)

plt.imshow(
    low_pass_image,
    cmap="gray"
)

plt.title("Low-Pass Result")
plt.axis("off")


plt.subplot(2, 3, 5)

plt.imshow(
    high_mask_display,
    cmap="gray"
)

plt.title("High-Pass Mask")
plt.axis("off")


plt.subplot(2, 3, 6)

plt.imshow(
    high_pass_image,
    cmap="gray"
)

plt.title("High-Pass Result")
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

print("\n======================================")
print("Experiment 4 completed successfully!")
print("======================================")

print("\nGenerated files:")

for file in sorted(os.listdir("output")):
    print("-", file)