import streamlit as st
import cv2
import numpy as np
from PIL import Image,ImageEnhance
import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D
import streamlit as st
from PIL import Image, ImageEnhance
import io
# Function for Blind Deconvolution
import numpy as np

def blind_deconvolution(image, psf, iterations=10):
    # Initialize the estimated image
    estimated_image = np.copy(image)

    # Normalize the point spread function (PSF)
    psf /= np.sum(psf)

    for _ in range(iterations):
        # Estimate the blurred image using the current estimated image and PSF
        blurred_image = convolve(estimated_image, psf)

        # Calculate the error between the observed image and the estimated image
        error = image / (blurred_image + 1e-10)

        # Update the estimated image using the error
        estimated_image *= convolve(error, np.flip(psf, axis=0), mode='constant')

    return estimated_image

# Function to perform 2D convolution
def convolve(image, kernel, mode='same'):
    return np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(kernel, s=image.shape)).real

# Example usage
# blurred_image = perform_blur(original_image, psf)
# estimated_image = blind_deconvolution(blurred_image, psf, iterations=10)

# Function for Dark Channel Prior
import numpy as np

def dark_channel_prior(image, window_size=15):
    # Convert the image to float
    image = image.astype(np.float64) / 255.0

    # Compute the dark channel of the image
    dark_channel = compute_dark_channel(image, window_size)

    # Estimate the atmospheric light
    atmospheric_light = estimate_atmospheric_light(image, dark_channel)

    # Estimate the transmission map
    transmission = estimate_transmission(image, atmospheric_light, window_size)

    # Recover the haze-free image
    recovered_image = recover_image(image, atmospheric_light, transmission)

    # Scale the recovered image back to 0-255 range
    recovered_image = (recovered_image * 255).astype(np.uint8)

    return recovered_image

def compute_dark_channel(image, window_size):
    min_channel = np.min(image, axis=2)
    return cv2.erode(min_channel, np.ones((window_size, window_size)))

def estimate_atmospheric_light(image, dark_channel, top_percent=0.001):
    pixel_count = dark_channel.size
    flat_dark_channel = dark_channel.flatten()
    flat_image = image.reshape((pixel_count, 3))

    dark_indices = np.argsort(-flat_dark_channel)
    top_idx = dark_indices[:int(pixel_count * top_percent)]

    return np.max(flat_image[top_idx], axis=0)

def estimate_transmission(image, atmospheric_light, window_size, omega=0.95):
    numerator = 1 - omega * compute_dark_channel(image / atmospheric_light, window_size)
    return 1 - numerator

def recover_image(image, atmospheric_light, transmission, t0=0.1):
    recovered_image = np.empty_like(image)
    for i in range(3):
        recovered_image[:, :, i] = ((image[:, :, i] - atmospheric_light[i]) / np.maximum(transmission, t0)) + atmospheric_light[i]
    return np.clip(recovered_image, 0, 1)

# Streamlit app
st.title('Image Enhancement using Blind Deconvolution and Dark Channel Prior')


# Function to adjust brightness
def adjust_brightness(image, factor):
    img = Image.open(image)
    enhancer = ImageEnhance.Brightness(img)
    bright_img = enhancer.enhance(factor)
    return bright_img

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display original image
    st.image(uploaded_image, caption='Original Image', use_column_width=True)

    # Adjust brightness
    brightness_factor = st.slider("Adjust brightness", 10.0, 15.0, 10.0, 0.1)
    bright_img = adjust_brightness(uploaded_image, brightness_factor)

    # Display adjusted image
    st.image(bright_img, caption='Enchanced Image', use_column_width=True)


import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


#***********************************************************

data_dir = "hazy/"

hazy = data_dir + 'hazy/'
Dehazy = data_dir + 'Dehazed/'


hazy_1 = os.listdir(hazy)
Dehazy_1 = os.listdir(Dehazy)


def loading(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (224, 224))
    return img[..., ::-1]


data = []
labels = []

for class_label in tqdm(os.listdir(data_dir)):
    class_path = os.path.join(data_dir, class_label)

    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        img = load_img(img_path, target_size=(224, 224))  # Resizing images to (224, 224)
        img_array = img_to_array(img)
        data.append(img_array)
        labels.append(class_label)

# Convert lists to numpy arrays
X = np.array(data)
y = np.array(labels)

# Convert class labels to numeric format
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# The number of unique classes
num_classes = len(label_encoder.classes_)

# Convert labels to one-hot encoding
y_onehot = to_categorical(y_encoded, num_classes=num_classes)

#***********************************************************

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=42)

# Build the CNN model 

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('model.h5')

# Plotting the training history
st.title("Plotting the Graph")
def plot_history(graph):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(graph.history['accuracy'])
    plt.plot(graph.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(graph.history['loss'])
    plt.plot(graph.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

plot_history(history)

plt.savefig("Model Accuracy")
plt.savefig("Model Loss")
plt.show()




# Evaluate the model
st.image("Model Accuracy.png")

loss, accuracy = model.evaluate(X_train, y_train)
st.title("CNN Algorithm")
st.write(f"Train Loss: {loss:.4f}")
st.write(f"Train Accuracy: {accuracy:.4f}")
val_loss, val_accuracy = model.evaluate(X_test, y_test)
st.write(val_accuracy*100)
