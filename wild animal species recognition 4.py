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
import winsound

# Download WordNet dataset
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

# Function to open live camera
def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
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
            winsound.Beep(1000, 500)  # Alert sound when an animal is detected
        cv2.imshow("Animal Species Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to enable mouse scroll
def on_mouse_wheel(event):
    canvas.yview_scroll(-1 * (event.delta // 120), "units")

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

# Function to exit the application
def exit_application():
    root.destroy()

def display_image(file_path, result_text):
    img = Image.open(file_path)
    img = img.resize((200, 200))  # Resize image for display
    img = ImageTk.PhotoImage(img)

    img_label = Label(image_frame, image=img, text=result_text, compound="top", fg="white", bg="#2c3e50", font=("Arial", 12))
    img_label.image = img  # Keep reference to avoid garbage collection
    img_label.pack(pady=10)

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
        detected_animals = []
        
        for pred in predictions:
            label, confidence = pred[1], pred[2] * 100
            detected_animals.append(f"{label} ({confidence:.2f}%)")
        
        result_text = "Detected: " + ", ".join(detected_animals) if detected_animals else "No animal detected"
        display_image(file_path, result_text)
        plot_graphs(predictions)

# Initialize Tkinter Window
root = tk.Tk()
root.title("Animal Detection System")
root.geometry("900x650")
root.configure(bg="#2c3e50")

# Buttons
upload_btn = Button(root, text="Upload Images", command=upload_images, font=("Arial", 14), bg="#2980b9", fg="white")
upload_btn.pack(pady=10)

camera_btn = Button(root, text="Open Camera", command=open_camera, font=("Arial", 14), bg="#2980b9", fg="white")
camera_btn.pack(pady=10)

exit_btn = Button(root, text="Exit", command=exit_application, font=("Arial", 14), bg="#c0392b", fg="white")
exit_btn.pack(pady=10)

# Create a scrollable frame
frame_container = Frame(root)
frame_container.pack(fill="both", expand=True)

canvas = Canvas(frame_container, bg="#2c3e50")
scrollbar = Scrollbar(frame_container, orient="vertical", command=canvas.yview)
scrollable_frame = Frame(canvas, bg="#2c3e50")

scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

canvas.bind_all("<MouseWheel>", on_mouse_wheel)  # Enable scrolling with mouse wheel

# Frame for displaying images
image_frame = Frame(scrollable_frame, bg="#2c3e50")
image_frame.pack()

# Frame for displaying images
image_frame = Frame(scrollable_frame, bg="#2c3e50")
image_frame.pack()

footer_label = Label(
    root,
    text="College: Govt Polytechnic M.H Halli\nDepartment: Computer Science and Engineering\nCreators: JAYANTH H P (189CS22021), YASHVANTH B D (189CS22056), MADAN H C (189CS23303)",
    font=("Arial", 8),
    fg="white",
    bg="#1e1e1e"
)
footer_label.pack(side="bottom", fill="x", pady=10)
footer_label.config(anchor="center")

# Run the GUI
root.mainloop()
