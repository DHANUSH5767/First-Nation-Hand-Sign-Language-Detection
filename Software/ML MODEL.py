import os
import cv2

dataset_path = r"C:\Users\mail4\PycharmProjects\Software\Dataset"
output_base = "processed_dataset"
rgb_folder = os.path.join(output_base, "RGB")
gray_folder = os.path.join(output_base, "Grayscale")
bw_folder = os.path.join(output_base, "BlackWhite")

for folder in [rgb_folder, gray_folder, bw_folder]:
    os.makedirs(folder, exist_ok=True)

for sign_name in os.listdir(dataset_path):
    sign_path = os.path.join(dataset_path, sign_name)
    if not os.path.isdir(sign_path) or sign_name.startswith('.'):
        continue

    for video_file in os.listdir(sign_path):
        if not video_file.lower().endswith(".mov"):
            continue

        video_path = os.path.join(sign_path, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (224, 224))

            rgb_dir = os.path.join(rgb_folder, sign_name)
            os.makedirs(rgb_dir, exist_ok=True)
            rgb_path = os.path.join(rgb_dir, f"{video_file[:-4]}_frame{frame_num}.jpg")
            cv2.imwrite(rgb_path, frame_resized)

            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            gray_dir = os.path.join(gray_folder, sign_name)
            os.makedirs(gray_dir, exist_ok=True)
            gray_path = os.path.join(gray_dir, f"{video_file[:-4]}_frame{frame_num}.jpg")
            cv2.imwrite(gray_path, gray_frame)

            _, bw_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
            bw_dir = os.path.join(bw_folder, sign_name)
            os.makedirs(bw_dir, exist_ok=True)
            bw_path = os.path.join(bw_dir, f"{video_file[:-4]}_frame{frame_num}.jpg")
            cv2.imwrite(bw_path, bw_frame)

            frame_num += 1

        cap.release()

print("✅ All video frames extracted and saved to RGB, Grayscale, and BlackWhite folders.")
