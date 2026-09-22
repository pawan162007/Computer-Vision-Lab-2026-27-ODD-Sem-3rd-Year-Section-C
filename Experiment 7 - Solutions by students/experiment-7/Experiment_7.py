import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time


# =========================================================
# 1. Create Output Folder
# =========================================================

os.makedirs("output", exist_ok=True)


# =========================================================
# 2. Load Input Video
# =========================================================

video_path = "input/input_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open input video!")
    exit()

print("Video loaded successfully.")


# =========================================================
# 3. Video Information
# =========================================================

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\nVIDEO INFORMATION")
print("--------------------------------")
print("Width        :", width)
print("Height       :", height)
print("FPS          :", fps)
print("Total Frames :", total_frames)


# =========================================================
# 4. Read First Frame
# =========================================================

ret, first_frame = cap.read()

if not ret:
    print("Error: Could not read video.")
    cap.release()
    exit()

old_gray = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2GRAY
)


# =========================================================
# 5. Lucas-Kanade Parameters
# =========================================================

feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)


# =========================================================
# 6. Detect Feature Points
# =========================================================

p0 = cv2.goodFeaturesToTrack(
    old_gray,
    mask=None,
    **feature_params
)

if p0 is None:
    print("No feature points detected!")
    cap.release()
    exit()

print("\nInitial Feature Points:", len(p0))


# =========================================================
# 7. Create Trajectory Mask
# =========================================================

lk_mask = np.zeros_like(first_frame)


# =========================================================
# 8. Prepare Output Videos
# =========================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

lk_output = cv2.VideoWriter(
    "output/lucas_kanade_output.mp4",
    fourcc,
    fps,
    (width, height)
)

dense_output = cv2.VideoWriter(
    "output/farneback_output.mp4",
    fourcc,
    fps,
    (width, height)
)


# =========================================================
# 9. Process Video Frames
# =========================================================

frame_count = 0

lk_times = []
farneback_times = []

last_lk_frame = first_frame.copy()
last_dense_frame = first_frame.copy()
last_magnitude = None


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # =====================================================
    # Lucas-Kanade Sparse Optical Flow
    # =====================================================

    start_time = time.time()

    p1, status, error = cv2.calcOpticalFlowPyrLK(
        old_gray,
        frame_gray,
        p0,
        None,
        **lk_params
    )

    lk_time = time.time() - start_time

    lk_times.append(lk_time)

    lk_frame = frame.copy()

    if p1 is not None:

        good_new = p1[status == 1]
        good_old = p0[status == 1]

        for new, old in zip(
            good_new,
            good_old
        ):

            a, b = new.ravel()
            c, d = old.ravel()

            a = int(a)
            b = int(b)
            c = int(c)
            d = int(d)

            # Draw trajectory
            lk_mask = cv2.line(
                lk_mask,
                (a, b),
                (c, d),
                (0, 255, 0),
                2
            )

            # Draw motion arrow
            lk_frame = cv2.arrowedLine(
                lk_frame,
                (c, d),
                (a, b),
                (0, 0, 255),
                2,
                tipLength=0.3
            )

            # Draw feature point
            lk_frame = cv2.circle(
                lk_frame,
                (a, b),
                4,
                (255, 0, 0),
                -1
            )

        lk_frame = cv2.add(
            lk_frame,
            lk_mask
        )

        p0 = good_new.reshape(
            -1,
            1,
            2
        )

    # Re-detect points if too few remain
    if p0 is None or len(p0) < 10:

        p0 = cv2.goodFeaturesToTrack(
            frame_gray,
            mask=None,
            **feature_params
        )

    last_lk_frame = lk_frame.copy()


    # =====================================================
    # Farneback Dense Optical Flow
    # =====================================================

    start_time = time.time()

    flow = cv2.calcOpticalFlowFarneback(
        old_gray,
        frame_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    farneback_time = time.time() - start_time

    farneback_times.append(
        farneback_time
    )


    # =====================================================
    # Convert Flow to HSV
    # =====================================================

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    hsv = np.zeros_like(frame)

    hsv[..., 1] = 255

    hsv[..., 0] = (
        angle * 180 / np.pi / 2
    ).astype(np.uint8)

    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    dense_frame = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    last_dense_frame = dense_frame.copy()
    last_magnitude = magnitude.copy()


    # =====================================================
    # Save Frames to Output Videos
    # =====================================================

    lk_output.write(lk_frame)
    dense_output.write(dense_frame)


    # =====================================================
    # Update Previous Frame
    # =====================================================

    old_gray = frame_gray.copy()

    frame_count += 1


# =========================================================
# 10. Release Resources
# =========================================================

cap.release()

lk_output.release()
dense_output.release()


# =========================================================
# 11. Save Final Images
# =========================================================

cv2.imwrite(
    "output/lucas_kanade_final.jpg",
    last_lk_frame
)

cv2.imwrite(
    "output/farneback_final.jpg",
    last_dense_frame
)


# =========================================================
# 12. Save Motion Magnitude
# =========================================================

if last_magnitude is not None:

    magnitude_image = cv2.normalize(
        last_magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    magnitude_image = np.uint8(
        magnitude_image
    )

    cv2.imwrite(
        "output/motion_magnitude.jpg",
        magnitude_image
    )


# =========================================================
# 13. Comparison Image
# =========================================================

first_rgb = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2RGB
)

lk_rgb = cv2.cvtColor(
    last_lk_frame,
    cv2.COLOR_BGR2RGB
)

dense_rgb = cv2.cvtColor(
    last_dense_frame,
    cv2.COLOR_BGR2RGB
)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(first_rgb)
plt.title("Original Frame")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(lk_rgb)
plt.title("Lucas-Kanade Sparse Optical Flow")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(dense_rgb)
plt.title("Farneback Dense Optical Flow")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "output/optical_flow_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# 14. Processing Time
# =========================================================

average_lk_time = np.mean(lk_times)

average_farneback_time = np.mean(
    farneback_times
)

average_motion = (
    np.mean(last_magnitude)
    if last_magnitude is not None
    else 0
)


# =========================================================
# 15. Final Results
# =========================================================

print("\n========================================")
print("OPTICAL FLOW RESULTS")
print("========================================")

print("Frames Processed       :", frame_count)

print(
    "Average LK Time        :",
    round(average_lk_time, 6),
    "seconds/frame"
)

print(
    "Average Farneback Time :",
    round(average_farneback_time, 6),
    "seconds/frame"
)

print(
    "Average Motion         :",
    round(float(average_motion), 4)
)

print("\n========================================")
print("Experiment 7 completed successfully!")
print("========================================")

print("\nGenerated files:")

for file in sorted(
    os.listdir("output")
):
    print("-", file)