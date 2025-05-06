Plains Indian Sign Language Hand Gesture Detection

Hybrid Machine Learning Model + Real-Time Hand Gesture Detection Software
Masters of Applied Computing | Wilfrid Laurier University
Completed: April 2025

Overview

This project presents a complete pipeline for recognizing hand gestures in Plains Indian Sign Language (PISL) using a hybrid deep learning model combined with real-time inference capabilities.

The primary objectives of the project were:

Preprocessing the dataset by extracting frames and hand landmarks from videos.

Training a hybrid model combining Convolutional Neural Networks (CNN) and hand landmarks.

Developing real-time prediction software capable of live gesture recognition using a webcam.

Documenting the entire process in a Jupyter Notebook for reproducibility.

A demonstration video showcasing the real-time prediction software is available under:

Software/Demonstration.mp4

Project Structure

Model/

Preprocessing and Landmark Extraction Scripts

Hybrid_hand_model.h5

Processed Dataset (Shortcut)

Software/

realtime_prediction.py

datasets/

preprocessed_dataset/

landmarks/

Demonstration.mp4

Notebook/

Plains_HandSign_Detection.ipynb

processed_dataset/

README.md

Technologies Used

Python 3.x

MediaPipe: Real-time hand tracking and landmark detection

OpenCV: Frame extraction and real-time video processing

TensorFlow / Keras: Hybrid machine learning model development

NumPy, Pandas: Data preprocessing

Jupyter Notebook: Experimentation and process documentation

Hybrid Model Architecture

The hybrid model integrates:

Convolutional layers to process RGB, Grayscale, and Binary image frames.

Landmark feature layers to process 21-point hand landmark vectors.

Fully connected layers to combine image and landmark features for final gesture classification.

The model achieved approximately 96% accuracy on the validation dataset.

Real-Time Gesture Prediction

The real-time prediction software (realtime_prediction.py) offers the following functionalities:

Captures live video feed from the webcam.

Detects hand landmarks in real time using MediaPipe.

Processes frames and landmarks through the trained hybrid model.

Displays predicted gestures with bounding boxes and annotations on the screen.

The software is ready-to-use. All necessary datasets and the trained model are included in the Software directory. No additional training is required.

Notebook for Training and Review

The Notebook/Plains_HandSign_Detection.ipynb file provides a step-by-step demonstration of:

Loading and preprocessing the dataset.

Extracting hand landmarks.

Building and training the hybrid model.

Evaluating model performance and saving the final model.

This notebook is designed for ease of review and to allow others to reproduce the experiment if required.

Running the Software

To run the real-time prediction software, execute the following commands:

cd Software
python realtime_prediction.py

Ensure that a webcam is connected and accessible for capturing live video input.

Acknowledgements

This project was completed under the guidance of [Professor’s Name], as part of the Masters of Applied Computing program at Wilfrid Laurier University.

Special thanks to the members of Group 3 for their collaborative effort in developing the dataset, training the model, and building the real-time application.

License

This project is intended for academic and portfolio use only. All data and code are provided for educational purposes.

