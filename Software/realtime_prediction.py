import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time

model = load_model("hybrid_hand_model.h5")
labels = ['10', '3', '6', 'Afternoon', 'Afterwards', 'Apple', 'Arrange', 'Bag', 'Between',
          'Bite', 'Book', 'Bowl', 'Boy', 'Different', 'Down', 'He', 'Love', 'Man', 'Opposite', 'Sick']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

CONFIDENCE_THRESHOLD = 0.85  # lowered slightly for better detection
current_label = ""

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    max_confidence = 0
    predicted_label = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_set = []
            for lm in hand_landmarks.landmark:
                landmark_set.extend([lm.x, lm.y, lm.z])

            if len(landmark_set) == 63:
                cropped = cv2.resize(frame, (128, 128))
                img_array = img_to_array(cropped).astype("float16") / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                landmark_array = np.array(landmark_set).reshape(1, 63).astype("float16")

                try:
                    predictions = model.predict([img_array, landmark_array], verbose=0)
                    class_id = np.argmax(predictions)
                    confidence = predictions[0][class_id]

                    print(f"🧠 Prediction: {labels[class_id]}, Confidence: {confidence:.2f}")  # Debug line

                    if confidence > CONFIDENCE_THRESHOLD and confidence > max_confidence:
                        max_confidence = confidence
                        predicted_label = labels[class_id]

                except Exception as e:
                    print("❌ Prediction error:", e)

    if predicted_label:
        current_label = predicted_label

    if current_label:
        cv2.putText(frame, current_label, (int(w / 2) - 100, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Real-Time Hand Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time

model = load_model("hybrid_hand_model.h5")
labels = ['10', '3', '6', 'Afternoon', 'Afterwards', 'Apple', 'Arrange', 'Bag', 'Between',
          'Bite', 'Book', 'Bowl', 'Boy', 'Different', 'Down', 'He', 'Love', 'Man', 'Opposite', 'Sick']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

CONFIDENCE_THRESHOLD = 0.85  # lowered slightly for better detection
current_label = ""

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    max_confidence = 0
    predicted_label = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_set = []
            for lm in hand_landmarks.landmark:
                landmark_set.extend([lm.x, lm.y, lm.z])

            if len(landmark_set) == 63:
                cropped = cv2.resize(frame, (128, 128))
                img_array = img_to_array(cropped).astype("float16") / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                landmark_array = np.array(landmark_set).reshape(1, 63).astype("float16")

                try:
                    predictions = model.predict([img_array, landmark_array], verbose=0)
                    class_id = np.argmax(predictions)
                    confidence = predictions[0][class_id]

                    print(f"🧠 Prediction: {labels[class_id]}, Confidence: {confidence:.2f}")  # Debug line

                    if confidence > CONFIDENCE_THRESHOLD and confidence > max_confidence:
                        max_confidence = confidence
                        predicted_label = labels[class_id]

                except Exception as e:
                    print("❌ Prediction error:", e)

    if predicted_label:
        current_label = predicted_label

    if current_label:
        cv2.putText(frame, current_label, (int(w / 2) - 100, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Real-Time Hand Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
