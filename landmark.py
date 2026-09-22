import cv2 as cv
import mediapipe as mp
import pandas as pd
import os

# Initialize mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Input/output folders
# For flat back
input_folder = "dataset/deadlift/back_angle/flat_back"
output_folder = "landmarks/back_angle/Correct"

# For rounded back
# input_folder = "dataset/deadlift/back_angle/rounded_back"
# output_folder = "landmarks/back_angle/Rounded"

# for correct knee
# input_folder = "dataset/deadlift/knees/Correct"
# output_folder = "landmarks/knee_valgus/Correct"

# for knee valgus
# input_folder = "dataset/deadlift/knees/knee_valgus"
# output_folder = "landmarks/knee_valgus/knee_cave"


os.makedirs(output_folder, exist_ok=True)

# Loop through all videos
for video_file in os.listdir(input_folder):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(input_folder, video_file)
        print(f" Processing: {video_file}")

        cap = cv.VideoCapture(video_path)
        landmark_list = []
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = pose.process(image)

            # Extract pose landmarks
            if results.pose_landmarks:
                landmarks = []
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                landmark_list.append([frame_num] + landmarks)

        cap.release()

        # Define column names
        columns = ["frame"]
        for i in range(33):
            columns += [f"x_{i}", f"y_{i}", f"z_{i}", f"vis_{i}"]

        # Save to CSV
        df = pd.DataFrame(landmark_list, columns=columns)
        csv_filename = os.path.join(output_folder, video_file.replace(".mp4", ".csv"))
        df.to_csv(csv_filename, index=False)

        print(f" Saved landmarks to: {csv_filename}\n")


