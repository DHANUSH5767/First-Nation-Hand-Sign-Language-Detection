✋ Plains Indian Sign Language Hand Gesture Detection

> 🧠 Hybrid ML Model + 🎥 Real-Time Hand Gesture Detection Software  
> [Masters of Applied Computing] | [Wilfrid Laurier University]  
> Completed: April 2025

---

## 🚀 Overview

This project demonstrates a **complete pipeline for recognizing hand gestures in Plains Indian Sign Language (PISL)** using a **hybrid deep learning model** with real-time inference capability.

The project covers:

- 📦 **Preprocessing dataset** → Extract frames + hand landmarks  
- 🧠 **Hybrid Model Training** → CNN + Hand Landmarks → `Hybrid_hand_model.h5`  
- 🎮 **Real-time Prediction Software** → Live gesture detection via webcam  
- 📒 **Jupyter Notebook** → Full walkthrough & reproducibility

A video demonstration is available under `Software/Demonstration.mp4`.

---

## 📁 Project Structure

├── Model/
│ ├── Preprocessing & Landmark Scripts
│ ├── Hybrid_hand_model.h5
│ ├── Processed Dataset (Shortcut)
│
├── Software/
│ ├── realtime_prediction.py
│ ├── datasets/
│ ├── preprocessed_dataset/
│ ├── landmarks/
│ ├── Demonstration.mp4
│
├── Notebook/
│ ├── Plains_HandSign_Detection.ipynb
│ ├── processed_dataset/
│
└── README.md

## 🧰 Technologies Used

- Python 3.x
- MediaPipe → Real-time hand tracking
- OpenCV → Frame extraction + real-time video processing
- TensorFlow / Keras → Hybrid ML model
- NumPy, Pandas → Data preprocessing
- Jupyter Notebook → Project walkthrough

---

## 🧠 Hybrid Model

Combines:
- 📷 **CNN layers** → RGB, Grayscale, Binary frames  
- ✋ **Landmark vectors** → 21 point hand keypoints  
- 🔗 **Fully Connected Layers (MLP)** → Final gesture classification

> **Achieved ~96% accuracy on validation set.**

---

## 🎮 Real-Time Gesture Prediction

`realtime_prediction.py` runs gesture detection live:

- Uses webcam feed
- Detects hand landmarks via MediaPipe
- Passes frame + landmarks to hybrid model
- Displays live predictions + bounding boxes

✅ Easy to run → All datasets + model included in `/Software`  
✅ Pre-trained model → No need to retrain

---

## 📒 Notebook for Training & Review

Notebook (`Notebook/Plains_HandSign_Detection.ipynb`) shows:

- Dataset loading and preprocessing
- Landmark extraction
- Model architecture
- Training and evaluation
- Final model exportyou can find a .ipynb file and the other relevant folders in the same. In the python notebook file, the executed process will be available for viewing and reviewing our project.

Run Command
cd Software
python realtime_prediction.py
