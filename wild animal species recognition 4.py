import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import nltk
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas, Scrollbar, Frame
from PIL import Image, ImageTk

# Download WordNet datas
nltk.download('wordnet')
nltk.download('omw-1.4')  # Optional: For additional languages

# Load Pre-Trained Model
print("Loading pre-trained MobileNetV2 model...")
model = MobileNetV2(weights="imagenet")
print("Model loaded successfully!")

# Function to check if the label is an animal
def is_animal(label):
    synsets = wordnet.synsets(label)
    for synset in synsets:
        if "animal" in synset.lexname():
            return True
    return False

# Function to predict species
def predict_species(image):
    image_resized = cv2.resize(image, (224, 224))
    input_data = np.expand_dims(image_resized, axis=0)
    input_data = preprocess_input(input_data)
    predictions = model.predict(input_data)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    filtered_predictions = [pred for pred in decoded_predictions if is_animal(pred[1])]
    return filtered_predictions

# Function to upload multiple images and detect animals
def upload_images():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_paths:
        return
    
    for widget in image_frame.winfo_children():
        widget.destroy()
    
    for file_path in file_paths:
        image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictions = predict_species(image)
        if predictions:
            label, confidence = predictions[0][1], predictions[0][2] * 100
            result_text = f"Detected: {label} ({confidence:.2f}%)"
        else:
            result_text = "No animal detected"
        
        display_image(file_path, result_text)
        plot_graphs(predictions)

# Function to open live camera
def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        result_label.config(text="Error: Could not access the webcam.", fg="red")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        predictions = predict_species(frame)
        if predictions:
            label, confidence = predictions[0][1], predictions[0][2] * 100
            cv2.putText(frame, f"{label} ({confidence:.2f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Animal Species Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to display images on UI
def display_image(file_path, result_text):
    frame = tk.Frame(image_frame, bg="black")
    frame.pack(pady=5)
    
    image = Image.open(file_path)
    image = image.resize((200, 200))
    photo = ImageTk.PhotoImage(image)
    img_label = Label(frame, image=photo, bg="black")
    img_label.image = photo
    img_label.pack()
    
    text_label = Label(frame, text=result_text, font=("Arial", 12), fg="white", bg="black")
    text_label.pack()

# Function to plot graphs
def plot_graphs(predictions):
    if not predictions:
        return
    labels = [pred[1] for pred in predictions]
    scores = [pred[2] * 100 for pred in predictions]
    loss = [100 - score for score in scores]
    epochs = list(range(1, len(scores) + 1))
    
    # Scatter Plot for Accuracy
    plt.figure(figsize=(6, 4))
    plt.scatter(epochs, scores, color='blue')
    plt.title("Accuracy - Scatter Plot")
    plt.xlabel("Prediction Rank")
    plt.ylabel("Confidence (%)")
    plt.show()
    
    # Histogram for Loss
    plt.figure(figsize=(6, 4))
    plt.hist(loss, bins=5, color='red', alpha=0.7)
    plt.title("Loss Histogram")
    plt.xlabel("Loss (%)")
    plt.ylabel("Frequency")
    plt.show()
    
    # Line Graph for Accuracy vs Loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, scores, marker='o', linestyle='-', color='blue', label='Accuracy')
    plt.plot(epochs, loss, marker='s', linestyle='--', color='red', label='Loss')
    plt.title("Accuracy vs. Loss")
    plt.xlabel("Prediction Rank")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.show()

# Initialize Tkinter Window
root = tk.Tk()
root.title("Animal Detection System")
root.geometry("800x600")
root.configure(bg="black")

# Create a canvas for scrolling
canvas = Canvas(root, bg="black")
scroll_y = Scrollbar(root, orient="vertical", command=canvas.yview)
frame = Frame(canvas, bg="black")
canvas.create_window((0, 0), window=frame, anchor="nw")
canvas.configure(yscrollcommand=scroll_y.set)

# Upload Button
upload_btn = Button(root, text="Upload Images", command=upload_images, font=("Arial", 14), bg="gray", fg="white")
upload_btn.pack(pady=10)

# Live Camera Button
camera_btn = Button(root, text="Open Camera", command=open_camera, font=("Arial", 14), bg="gray", fg="white")
camera_btn.pack(pady=10)

# Frame for displaying images
image_frame = Frame(frame, bg="black")
image_frame.pack()

# Result Label
result_label = Label(root, text="", font=("Arial", 16), fg="white", bg="black")
result_label.pack(pady=10)

canvas.pack(side="left", fill="both", expand=True)
scroll_y.pack(side="right", fill="y")
frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Run the GUI
root.mainloop()
