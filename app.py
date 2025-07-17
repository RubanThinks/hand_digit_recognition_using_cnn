import streamlit as st
st.set_page_config(page_title="Draw a Digit", layout="centered")

from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image


# Load model

from tensorflow import keras


st.title("✍️ Draw a Digit (0–9)")

# ✅ Recreate SAME architecture (must match Colab training!)
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(28,28,1)),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="sigmoid"),
    keras.layers.Dense(10, activation="softmax")
])
model.summary()
from tensorflow.keras.models import load_model

# Load the ENTIRE model (architecture + weights)
model2= load_model("final.h5")

# Now you can continue with your Streamlit UI logic...


st.markdown("Draw your digit in the box below and click Predict!")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="#000000",  # Black background
    stroke_width=15,
    stroke_color="#FFFFFF",  # White stroke
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Show the drawn image
    st.image(canvas_result.image_data, caption="Your Drawing", width=150)

    if st.button("Predict"):
        # Process the image for model input
        img = canvas_result.image_data[:, :, 0]  # Use grayscale only
        img = Image.fromarray(img).resize((28, 28)).convert("L")
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model2.predict(img_array)
        predicted_digit = np.argmax(prediction)

        st.success(f"Predicted Digit: {predicted_digit}")
