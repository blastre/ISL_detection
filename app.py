import streamlit as st
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import numpy as np

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load your saved GRU model
model = load_model('gru_sign_language_model.h5')

# Define labels in English and Gujarati
labels = ["amitabh bachchan", "enjoy", "pizza"]
labels_gujarati = ["અમિતાભ બચ્ચન", "આનંદ માણો", "પિઝા"]

# Streamlit interface
st.title('Sign Language Detection')
confidence_threshold = st.slider('Confidence Threshold', min_value=0.0, max_value=1.0, value=0.65, step=0.01)

def extract_keypoints(results):
    def landmarks_to_array(landmarks):
        if landmarks:
            return np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks]).flatten()
        return np.zeros(63)  # Adjust size based on landmarks

    pose_landmarks = landmarks_to_array(results.pose_landmarks.landmark if results.pose_landmarks else [])
    left_hand_landmarks = landmarks_to_array(results.left_hand_landmarks.landmark if results.left_hand_landmarks else [])
    right_hand_landmarks = landmarks_to_array(results.right_hand_landmarks.landmark if results.right_hand_landmarks else [])

    keypoints = np.concatenate([pose_landmarks, left_hand_landmarks, right_hand_landmarks])
    return keypoints

# Set up webcam
video_capture = cv2.VideoCapture(0)
holistic = mp_holistic.Holistic()
sequence = []

stframe = st.empty()
text_placeholder = st.empty()  # Placeholder for the text

# Initialize prediction label
prediction_label = "Unknown"
prediction_label_gujarati = "અજ્ઞાત"

while True:
    ret, frame = video_capture.read()
    if not ret:
        st.write("Failed to grab frame.")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)

    if len(sequence) == 10:
        sequence_np = np.expand_dims(sequence, axis=0)
        prediction_probs = model.predict(sequence_np)[0]

        # Find the most confident prediction
        max_index = np.argmax(prediction_probs)
        max_confidence = prediction_probs[max_index]

        if max_confidence >= confidence_threshold:
            prediction_label = labels[max_index]
            prediction_label_gujarati = labels_gujarati[max_index]
        else:
            prediction_label = "Unknown"
            prediction_label_gujarati = "અજ્ઞાત"

        # Reset sequence after making prediction
        sequence = []

    # Draw landmarks on the frame for visualization
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Display the frame (without the text)
    stframe.image(frame, channels="BGR")

    # Update the prediction text in the placeholder (overwrites previous text)
    text_placeholder.subheader(f"Prediction: {prediction_label} ({prediction_label_gujarati})")

video_capture.release()
