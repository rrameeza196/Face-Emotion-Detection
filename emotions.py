import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import cv2
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Emotion label dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
                4: "Neutral", 5: "Sad", 6: "Surprised"}

# Build CNN model
def build_model():
    model = Sequential()
    model.add(Input(shape=(48, 48, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(62, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model

# Plot training 
def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(model_history.history['accuracy'])
    axs[0].plot(model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(model_history.history['loss'])
    axs[1].plot(model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')

    st.pyplot(fig)

# Model training
def train_model():
    st.subheader("📊 Model Training")
    with st.spinner('Training the model...'):

        train_dir = 'data/train'
        val_dir = 'data/test'
        num_train = 28709
        num_val = 7178
        batch_size = 64
        num_epoch = 10

        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir, target_size=(48, 48), batch_size=batch_size,
            color_mode="grayscale", class_mode='categorical')

        validation_generator = val_datagen.flow_from_directory(
            val_dir, target_size=(48, 48), batch_size=batch_size,
            color_mode="grayscale", class_mode='categorical')

        model = build_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.0001, decay=1e-6),
                      metrics=['accuracy'])

        history = model.fit(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)

        model.save('model.h5')  # Save full model
        st.success('✅ Model trained and saved as model.h5')
        plot_model_history(history)

# Real-time emotion detection
def display_emotion_detection():
    st.subheader("😊 Real-Time Facial Emotion Detection")

    try:
        model = load_model("model.h5")  # Load entire model
    except Exception as e:
        st.error("❌ Model file not found. Please train the model first.")
        return

    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to capture frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            if roi_gray.shape[0] == 0 or roi_gray.shape[1] == 0:
                continue
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cropped_img = cropped_img / 255.0
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            emotion = emotion_dict[maxindex]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 204, 0), 2)
            cv2.rectangle(frame, (x, y - 35), (x + w, y), (255, 204, 0), -1)
            cv2.putText(frame, emotion, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    cap.release()

# Streamlit App UI
st.title("🎭 Facial Emotion Recognition App")
option = st.sidebar.selectbox("Select Mode", ["Train Model", "Detect Emotions"])

if option == "Train Model":
    train_model()
elif option == "Detect Emotions":
    display_emotion_detection()
