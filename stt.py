import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load your saved GRU model
print("Loading model...")
model = load_model('gru_sign_language_model.h5')
print("Model loaded.")

def extract_keypoints(results):
    def landmarks_to_array(landmarks):
        if landmarks:
            return np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks]).flatten()
        return np.zeros(63)  # Adjust size based on landmarks (pose: 33*3, hands: 21*3 each)

    pose_landmarks = landmarks_to_array(results.pose_landmarks.landmark if results.pose_landmarks else [])
    left_hand_landmarks = landmarks_to_array(results.left_hand_landmarks.landmark if results.left_hand_landmarks else [])
    right_hand_landmarks = landmarks_to_array(results.right_hand_landmarks.landmark if results.right_hand_landmarks else [])

    keypoints = np.concatenate([pose_landmarks, left_hand_landmarks, right_hand_landmarks])
    return keypoints

def live_detection_from_webcam(sequence_length=10):
    cap = cv2.VideoCapture(0)
    holistic = mp_holistic.Holistic()
    sequence = []  # To store a sequence of keypoints
    threshold = 0.5  # Confidence threshold for classification

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert the frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        if results.pose_landmarks:
            print("Pose landmarks detected.")
        if results.left_hand_landmarks:
            print("Left hand landmarks detected.")
        if results.right_hand_landmarks:
            print("Right hand landmarks detected.")

        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # Limit the sequence to the specified length (10 frames)
        if len(sequence) == sequence_length:
            print("Making prediction...")
            sequence_np = np.expand_dims(sequence, axis=0)  # Reshape for model input
            prediction = model.predict(sequence_np)[0]
            print(f"Prediction: {prediction}")

            sequence = []

        # Draw landmarks on the frame for visualization
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('Sign Language Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Run live detection from the local webcam
live_detection_from_webcam()
