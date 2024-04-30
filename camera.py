import cv2
import mediapipe as mp
import numpy as np

# Initialize the MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the default camera
cap = cv2.VideoCapture(0)
thumb_coordinates = []
index_coordinates = []

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the hand landmarks
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert the coordinates to pixel values
            thumb_coords = np.array([thumb_tip.x * frame.shape[1], thumb_tip.y * frame.shape[0], thumb_tip.z],dtype=np.int64)
            index_coords = np.array([index_tip.x * frame.shape[1], index_tip.y * frame.shape[0], index_tip.z],dtype=np.int64)

            # Append the coordinates to the lists
            thumb_coordinates.append([thumb_coords[0], thumb_coords[1], thumb_coords[2]])
            index_coordinates.append([index_coords[0], index_coords[1], index_coords[2]])

            # Draw a circle at the thumb and index finger tips
            cv2.circle(frame, (thumb_coords[0], thumb_coords[1]), 10, (0, 0, 255), -1)
            cv2.circle(frame, (index_coords[0], index_coords[1]), 10, (0, 0, 255), -1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy the window
cap.release()
cv2.destroyAllWindows()

thumb_coordinates = np.array(thumb_coordinates)
index_coordinates = np.array(index_coordinates)

# Initialize lists to store velocities, accelerations and jerk
thumb_velocities = []
index_velocities = []
thumb_accelerations = []
index_accelerations = []
thumb_jerk = []
index_jerk = []

# Calculate velocities, accelerations and jerk for thumb and index coordinates
for coords in [thumb_coordinates, index_coordinates]:
    # Calculate velocities
    velocities = np.diff(coords, axis=0)
    # Calculate accelerations
    accelerations = np.diff(velocities, axis=0)
    # Calculate jerk
    jerk = np.diff(accelerations, axis=0)

    if coords is thumb_coordinates:
        thumb_velocities.append(velocities)
        thumb_accelerations.append(accelerations)
        thumb_jerk.append(jerk)
    else:
        index_velocities.append(velocities)
        index_accelerations.append(accelerations)
        index_jerk.append(jerk)

# Calculate mean velocities, accelerations and jerk
thumb_mean_velocity = np.mean(thumb_velocities, axis=0)
index_mean_velocity = np.mean(index_velocities, axis=0)
thumb_mean_acceleration = np.mean(thumb_accelerations, axis=0)
index_mean_acceleration = np.mean(index_accelerations, axis=0)
thumb_mean_jerk = np.mean(thumb_jerk, axis=0)
index_mean_jerk = np.mean(index_jerk, axis=0)

new_features=[]
new_features.append([np.mean(thumb_mean_velocity), np.mean(thumb_mean_acceleration), np.mean(thumb_mean_jerk), np.mean(index_mean_velocity), np.mean(index_mean_acceleration), np.mean(index_mean_jerk)])
new_features = np.array(new_features)

output_file = "new_features.txt"
np.savetxt(output_file, new_features)