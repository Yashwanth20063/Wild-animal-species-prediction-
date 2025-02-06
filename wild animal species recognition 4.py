import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import nltk
from nltk.corpus import wordnet
import matplotlib.pyplot as plt

# Download WordNet datas
nltk.download('wordnet')
nltk.download('omw-1.4')  # Optional: For additional languages

# Load Pre-Trained Model
print("Loading pre-trained MobileNetV2 model...")
model = MobileNetV2(weights="imagenet")
print("Model loaded successfully!")

# Define a function to check if the label corresponds to an animal
def is_animal(label):
    """Check if the label corresponds to an animal using WordNet."""
    synsets = wordnet.synsets(label)
    for synset in synsets:
        if "animal" in synset.lexname():
            return True
    return False

# Define a function to predict and decode species
def predict_species(image):
    image_resized = cv2.resize(image, (224, 224))
    input_data = np.expand_dims(image_resized, axis=0)
    input_data = preprocess_input(input_data)
    predictions = model.predict(input_data)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    filtered_predictions = [pred for pred in decoded_predictions if is_animal(pred[1])]
    return filtered_predictions

# Function to plot graphs
def plot_predictions(predictions):
    labels = [pred[1] for pred in predictions]
    scores = [pred[2] * 100 for pred in predictions]

    # Bar Chart
    plt.figure(figsize=(6, 4))
    plt.bar(labels, scores, color=['blue', 'green', 'orange'])
    plt.title("Accuracy - Bar Chart")
    plt.ylabel("Confidence (%)")
    plt.xlabel("Labels")
    plt.ylim(0, 100)
    for i, score in enumerate(scores):
        plt.text(i, score + 1, f"{score:.2f}%", ha='center')
    plt.show()

    # Pie Chart
    plt.figure(figsize=(6, 4))
    plt.pie(scores, labels=labels, autopct='%1.1f%%', startangle=90, colors=['blue', 'green', 'orange'])
    plt.title("Accuracy Distribution - Pie Chart")
    plt.show()

    # Line Graph: Accuracy vs. Loss (Example values for visualization)
    epochs = list(range(1, len(scores) + 1))
    loss = [100 - score for score in scores]
    
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, scores, marker='o', linestyle='-', color='blue', label='Accuracy')
    plt.plot(epochs, loss, marker='s', linestyle='--', color='red', label='Loss')
    plt.title("Accuracy vs. Loss - Line Graph")
    plt.xlabel("Prediction Rank (1 = Top Prediction)")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.show()

# Step 2: Upload an Image or Capture Live Video
print("Choose an option:\n1. Upload a photo\n2. Use live video (press 'q' to quit)")
choice = input("Enter your choice (1/2): ")

if choice == "1":
    print("Please upload a photo.")
    Tk().withdraw()
    image_path = askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

    if not image_path:
        print("No image selected. Exiting...")
        exit()

    image = cv2.imread(image_path)
    decoded_predictions = predict_species(image)

    if not decoded_predictions:
        print("Error: The uploaded image does not contain an animal.")
    else:
        print("Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            print(f"{i + 1}: {label} ({score * 100:.2f}%)")
        plot_predictions(decoded_predictions)
        top_prediction = decoded_predictions[0]
        label = top_prediction[1]
        confidence = top_prediction[2] * 100
        cv2.putText(image, f"{label} ({confidence:.2f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Predicted Species", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

elif choice == "2":
    print("Starting live video. Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        decoded_predictions = predict_species(frame)
        if decoded_predictions:
            top_prediction = decoded_predictions[0]
            label = top_prediction[1]
            confidence = top_prediction[2] * 100
            cv2.putText(frame, f"{label} ({confidence:.2f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not an animal", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Animal Species Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    print("Invalid choice. Exiting...")
    exit()
