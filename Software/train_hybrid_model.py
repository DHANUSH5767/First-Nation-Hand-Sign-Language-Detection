import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

print("📥 Loading landmark CSV file...")
df = pd.read_csv("hand_landmarks.csv")

print("🧪 Encoding labels...")
le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["label"])

print("🖼️ Loading RGB images (128x128)...")
def load_rgb_image(img_path):
    img = load_img(img_path, target_size=(128, 128))
    return img_to_array(img).astype("float16") / 255.0  # Use float16 to save memory

X_img = np.array([load_rgb_image(p) for p in df["image_path"]], dtype="float16")
print(f"✅ Loaded {len(X_img)} images")

print("📐 Extracting hand landmark features...")
landmark_cols = [col for col in df.columns if col.startswith(('x', 'y', 'z')) and col[1:].isdigit()]
X_landmark = df[landmark_cols].values.astype("float16")  # Also convert to float16
print(f"✅ Extracted {X_landmark.shape[1]} landmark features per sample")

print("🎯 One-hot encoding labels...")
y = tf.keras.utils.to_categorical(df["label_enc"], num_classes=len(le.classes_))

print("🧪 Splitting into training and validation sets...")
X_img_train, X_img_val, X_lm_train, X_lm_val, y_train, y_val = train_test_split(
    X_img, X_landmark, y, test_size=0.2, random_state=42
)
print(f"🔹 Training samples: {len(X_img_train)} | Validation samples: {len(X_img_val)}")

print("🧠 Building hybrid model architecture...")
input_img = Input(shape=(128, 128, 3))
cnn_base = MobileNetV2(include_top=False, input_tensor=input_img, weights="imagenet")
cnn_base.trainable = False
cnn_out = GlobalAveragePooling2D()(cnn_base.output)

input_lm = Input(shape=(63,))
lm_out = Dense(128, activation="relu")(input_lm)
lm_out = Dense(64, activation="relu")(lm_out)

combined = Concatenate()([cnn_out, lm_out])
x = Dense(128, activation="relu")(combined)
output = Dense(len(le.classes_), activation="softmax")(x)

model = Model(inputs=[input_img, input_lm], outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("🚀 Starting training...")
model.fit(
    [X_img_train, X_lm_train],
    y_train,
    validation_data=([X_img_val, X_lm_val], y_val),
    epochs=25,
    batch_size=32
)

print("💾 Saving trained model to 'hybrid_hand_model.h5'...")
model.save("hybrid_hand_model.h5")

print("✅ Model training complete & saved as hybrid_hand_model.h5")
