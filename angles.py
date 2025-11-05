import pandas as pd
import numpy as np
import os
import cv2
input_folder ="landmarks/back_angle/Correct"


# df =pd.read_csv("landmarks/back_angle/Correct/cb1.csv")
# video_path ="dataset/deadlift/back_angle/flat_back/cb1.mp4"
df =pd.read_csv("landmarks/back_angle/Rounded/rb2.csv")
video_path ="dataset/deadlift/back_angle/rounded_back/rb2.mp4"
shoulder = df[['x_11', 'y_11']]
hip = df[['x_23','y_23']]
knee = df[['x_25','y_25']]

a = shoulder
b = hip
c = knee
def calculate_angle(a,b,c):
     ba = np.array(a) - np.array(b)
     bc = np.array(c) - np.array(b)
     
     cosine_angle = np.dot(ba,bc)/ (np.linalg.norm(ba) * (np.linalg.norm(bc)))
     angle = np.degrees(np.arccos(cosine_angle))
     return angle
 
df['back_angle'] = df.apply(
    lambda row : calculate_angle(
        (row['x_11'], row['y_11']),
        (row['x_23'], row['y_23']),
        (row['x_25'], row['y_25'])
    ),
    axis=1
)

#  Calculate dy (change in y) for hip and shoulder
df['dy_shoulder'] = -df['y_11'].diff()
df['dy_hip'] = -df['y_23'].diff()

#  Set movement threshold
threshold = 0.003
# point where hip and shoulder start moving upward
hip_lifts = df.index[df['dy_hip'] > threshold].tolist()
shoulder_lifts = df.index[df['dy_shoulder'] > threshold].tolist()


# Grouping frames into reps
rep_frames = []
for i in range(len(hip_lifts) - 1):
    if hip_lifts[i+1] - hip_lifts[i] > 10: 
        rep_frames.append(hip_lifts[i])

# Ensure at least one rep is captured
if len(rep_frames) == 0:
    rep_frames = [hip_lifts[0]]
else:
    # Add last hip frame as end of final rep
    rep_frames.append(hip_lifts[-1])

# Store results for each rep
flagged_frames = []

# Detect form per repetition instead of per frame
for i in range(len(rep_frames) - 1):
    start = rep_frames[i]
    end = rep_frames[i + 1]

    # Extract frame segment for this rep
    hip_segment = df['dy_hip'][start:end]
    shoulder_segment = df['dy_shoulder'][start:end]

    # Find the frame where each moves the most
    hip_peak = hip_segment.idxmax()
    shoulder_peak = shoulder_segment.idxmax()

    # Compare who moves first
    if hip_peak < shoulder_peak:
        flagged_frames.append((hip_peak, "Rounded Back", (0, 0, 255)))
    else:
        flagged_frames.append((shoulder_peak, "Good Form", (0, 255, 0)))


print("Detected Lifts:")
for f, label, _ in flagged_frames:
    print(f" → Frame {f}: {label}")



cap = cv2.VideoCapture(video_path)
frame_index = 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_index >= len(df):
        break

    # Extract coordinates from current frame (normalized 0–1)
    x_shoulder, y_shoulder = df.loc[frame_index, ['x_11', 'y_11']]
    x_hip, y_hip = df.loc[frame_index, ['x_23', 'y_23']]
    x_knee, y_knee = df.loc[frame_index, ['x_25', 'y_25']]

    # Convert normalized to pixel coordinates
    h, w, _ = frame.shape
    shoulder = (int(x_shoulder * w), int(y_shoulder * h))
    hip = (int(x_hip * w), int(y_hip * h))
    knee = (int(x_knee * w), int(y_knee * h))

  
    angle = calculate_angle(shoulder, hip, knee)



 
    cv2.line(frame, shoulder, hip, (0, 255, 0), 3)
    cv2.line(frame, hip, knee, (0, 255, 0), 3)
    cv2.circle(frame, shoulder, 6, (0, 0, 255), -1)
    cv2.circle(frame, hip, 6, (255, 0, 0), -1)
    cv2.circle(frame, knee, 6, (0, 0, 255), -1)

  
    cv2.putText(frame, f"Back Angle: {int(angle)} deg", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display only one label per frame (avoid overlap)
    active_label = None
    active_color = (255, 255, 255)

    for flagged_frame, lift_label, lift_color in flagged_frames:
        if flagged_frame <= frame_index <= flagged_frame + 10:
            active_label = lift_label
            active_color = lift_color
            break  

    if active_label:
        cv2.putText(frame, active_label, (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, active_color, 3)


  
    cv2.imshow("Back Angle Visualization", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()