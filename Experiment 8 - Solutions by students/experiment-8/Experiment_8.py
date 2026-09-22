import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

os.makedirs("output", exist_ok=True)

video_path = "input/input_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open input video!")
    exit()

print("Video loaded successfully.")

fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print("\nVIDEO INFORMATION")
print("--------------------------------")
print("Width        :", width)
print("Height       :", height)
print("FPS          :", fps)
print("Total Frames :", total_frames)


ret, first_frame = cap.read()

if not ret:
    print("Error: Could not read video.")
    cap.release()
    exit()

old_gray = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2GRAY
)


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


p0 = cv2.goodFeaturesToTrack(
    old_gray,
    mask=None,
    **feature_params
)

if p0 is None:
    print("No feature points detected.")
    cap.release()
    exit()

print(
    "\nInitial Shi-Tomasi Points:",
    len(p0)
)



trajectory_mask = np.zeros_like(
    first_frame
)



fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

lk_writer = cv2.VideoWriter(
    "output/lucas_kanade_tracking.mp4",
    fourcc,
    fps,
    (width, height)
)

farneback_writer = cv2.VideoWriter(
    "output/farneback_motion_analysis.mp4",
    fourcc,
    fps,
    (width, height)
)



frame_count = 0

total_displacement = 0.0

displacements = []

directions = []

speed_values = []

last_lk_frame = first_frame.copy()

last_farneback_frame = first_frame.copy()

last_flow_magnitude = None


while True:

    ret, frame = cap.read()

    if not ret:
        break


    frame_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )



    p1, status, error = cv2.calcOpticalFlowPyrLK(
        old_gray,
        frame_gray,
        p0,
        None,
        **lk_params
    )


    tracking_frame = frame.copy()


    if p1 is not None:

        good_new = p1[status == 1]

        good_old = p0[status == 1]

        if len(good_new) > 0:

            movement_x = (
                good_new[:, 0] -
                good_old[:, 0]
            )

            movement_y = (
                good_new[:, 1] -
                good_old[:, 1]
            )

            distances = np.sqrt(
                movement_x ** 2 +
                movement_y ** 2
            )

            average_displacement = np.mean(
                distances
            )

            total_displacement += (
                average_displacement
            )

            displacements.append(
                average_displacement
            )

            avg_x = np.mean(
                movement_x
            )

            avg_y = np.mean(
                movement_y
            )

            angle = math.degrees(
                math.atan2(
                    avg_y,
                    avg_x
                )
            )

            directions.append(angle)



            if fps > 0:

                speed = (
                    average_displacement *
                    fps
                )

                speed_values.append(
                    speed
                )

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


                # Trajectory
                trajectory_mask = cv2.line(
                    trajectory_mask,
                    (a, b),
                    (c, d),
                    (0, 255, 0),
                    2
                )


                # Motion arrow
                tracking_frame = cv2.arrowedLine(
                    tracking_frame,
                    (c, d),
                    (a, b),
                    (0, 0, 255),
                    2,
                    tipLength=0.3
                )


                # Feature point
                tracking_frame = cv2.circle(
                    tracking_frame,
                    (a, b),
                    4,
                    (255, 0, 0),
                    -1
                )


            # Add trajectory
            tracking_frame = cv2.add(
                tracking_frame,
                trajectory_mask
            )


            cv2.putText(
                tracking_frame,
                f"Tracked Points: {len(good_new)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                tracking_frame,
                f"Displacement: {average_displacement:.2f} px",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                tracking_frame,
                f"Direction: {angle:.2f} deg",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                tracking_frame,
                f"Speed: {speed_values[-1]:.2f} px/s",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


            # Update points
            p0 = good_new.reshape(
                -1,
                1,
                2
            )

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


    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    last_flow_magnitude = magnitude.copy()

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


    farneback_frame = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


    # Add text
    cv2.putText(
        farneback_frame,
        "Farneback Dense Optical Flow",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    lk_writer.write(
        tracking_frame
    )

    farneback_writer.write(
        farneback_frame
    )


    last_lk_frame = tracking_frame.copy()

    last_farneback_frame = (
        farneback_frame.copy()
    )

    old_gray = frame_gray.copy()


    # Re-detect points if required
    if p0 is None or len(p0) < 10:

        p0 = cv2.goodFeaturesToTrack(
            old_gray,
            mask=None,
            **feature_params
        )


    frame_count += 1


cap.release()

lk_writer.release()

farneback_writer.release()


cv2.imwrite(
    "output/lucas_kanade_tracking_final.jpg",
    last_lk_frame
)

cv2.imwrite(
    "output/farneback_final.jpg",
    last_farneback_frame
)

if last_flow_magnitude is not None:

    magnitude_image = cv2.normalize(
        last_flow_magnitude,
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

if len(displacements) > 0:

    average_displacement = np.mean(
        displacements
    )

else:

    average_displacement = 0


if len(speed_values) > 0:

    average_speed = np.mean(
        speed_values
    )

else:

    average_speed = 0


if len(directions) > 0:

    average_direction = np.mean(
        directions
    )

else:

    average_direction = 0



plt.figure(figsize=(10, 5))

plt.plot(
    displacements
)

plt.title(
    "Object Displacement Across Frames"
)

plt.xlabel(
    "Frame Number"
)

plt.ylabel(
    "Displacement (pixels)"
)

plt.grid(True)

plt.savefig(
    "output/displacement_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    speed_values
)

plt.title(
    "Estimated Object Motion Speed"
)

plt.xlabel(
    "Frame Number"
)

plt.ylabel(
    "Speed (pixels/second)"
)

plt.grid(True)

plt.savefig(
    "output/speed_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


first_rgb = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2RGB
)

lk_rgb = cv2.cvtColor(
    last_lk_frame,
    cv2.COLOR_BGR2RGB
)

farneback_rgb = cv2.cvtColor(
    last_farneback_frame,
    cv2.COLOR_BGR2RGB
)


plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)

plt.imshow(first_rgb)

plt.title(
    "Original Frame"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(lk_rgb)

plt.title(
    "Lucas-Kanade Tracking"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(farneback_rgb)

plt.title(
    "Farneback Motion"
)

plt.axis("off")


plt.tight_layout()

plt.savefig(
    "output/final_tracking_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n==========================================")
print("REAL-TIME OBJECT TRACKING RESULTS")
print("==========================================")

print(
    "Frames Processed      :",
    frame_count
)

print(
    "Average Displacement  :",
    round(float(average_displacement), 2),
    "pixels/frame"
)

print(
    "Average Speed         :",
    round(float(average_speed), 2),
    "pixels/second"
)

print(
    "Average Direction     :",
    round(float(average_direction), 2),
    "degrees"
)


print("\n==========================================")
print("Experiment 8 completed successfully!")
print("==========================================")



print("\nGenerated files:")

for file in sorted(
    os.listdir("output")
):

    print("-", file)