import os
import cv2
import mediapipe as mp
import pandas as pd

# Define all three input folders
base_paths = {
    "RGB": "processed_dataset/RGB",
    "Grayscale": "processed_dataset/Grayscale",
    "BlackWhite": "processed_dataset/BlackWhite"
}

output_csv = "hand_landmarks.csv"
data = []

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
print("🔍 Starting landmark extraction...")

# Loop through each image type folder
for img_type, base_path in base_paths.items():
    print(f"📂 Processing {img_type} images from {base_path}")

    for label in os.listdir(base_path):
        label_folder = os.path.join(base_path, label)
        if not os.path.isdir(label_folder) or label.startswith('.'):
            continue

        for file in os.listdir(label_folder):
            if not file.lower().endswith(".jpg"):
                continue

            file_path = os.path.join(label_folder, file)
            img = cv2.imread(file_path)

            # Convert grayscale/BW to BGR for consistency
            if len(img.shape) < 3 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                row = []
                for lm in landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                row += [label, file_path, img_type]
                data.append(row)

hands.close()

# Save results to CSV
columns = [f'{c}{i}' for i in range(21) for c in ['x', 'y', 'z']] + ['label', 'image_path', 'image_type']
df = pd.DataFrame(data, columns=columns)
df.to_csv(output_csv, index=False)

print(f"✅ Extracted {len(df)} landmark rows from all image types and saved to {output_csv}")
