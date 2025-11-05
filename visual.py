import cv2 as cv

import mediapipe as mp

# Initialize mediapipe pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Load your video
# path = "dataset/deadlift/back_angle/flat_back/cb1.mp4"
# path = "dataset/deadlift/back_angle/rounded_back/rb3.mp4"
# path = "dataset/deadlift/knees/Correct/ck1.mp4"
path = "dataset/deadlift/knees/knee_valgus/kv1.mp4"

cap = cv.VideoCapture(path)



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

   

   
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(image)

   
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # Draw pose landmarks if detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        )
    
    height, width = image.shape[:2]
    new_width = 1000 
    new_height= 600
    scale = new_width / width
    scale = new_height / height
    image = cv.resize(image, (int(width * scale), int(height * scale)))


    cv.imshow("Video", image)

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

