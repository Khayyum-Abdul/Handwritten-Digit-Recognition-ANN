
import os
import streamlit as st
import numpy as np
import tensorflow as tf

from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Digit Recognition ANN",
    page_icon="🔢",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "model",
        "handwritten_digit_ann.keras"
    )
)

@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("Handwritten Digit Recognition")
st.write("Draw a digit between 0 and 9.")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
    return_image_data=True
)


def preprocess_image(image_data):
    img = Image.fromarray(
        image_data.astype("uint8")
    ).convert("RGBA")

    background = Image.new(
        "RGBA",
        img.size,
        (0, 0, 0, 255)
    )

    img = Image.alpha_composite(
        background,
        img
    ).convert("L")

    img_array = np.array(img)

    coords = np.argwhere(img_array > 30)

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img.crop(
        (x_min, y_min, x_max + 1, y_max + 1)
    )

    cropped.thumbnail((20, 20))

    new_img = Image.new("L", (28, 28), 0)

    x_offset = (28 - cropped.width) // 2
    y_offset = (28 - cropped.height) // 2

    new_img.paste(
        cropped,
        (x_offset, y_offset)
    )

    img_array = (
        np.array(new_img).astype("float32") / 255.0
    )

    return img_array.reshape(1, 28, 28)


if st.button("Predict Digit", type="primary"):

    if canvas_result.image_data is not None:

        img_array = preprocess_image(
            canvas_result.image_data
        )

        if img_array is None:
            st.warning("Please draw a digit first.")

        else:
            prediction = model.predict(
                img_array,
                verbose=0
            )

            predicted_digit = int(
                np.argmax(prediction)
            )

            confidence = (
                float(np.max(prediction)) * 100
            )

            st.success(
                f"Predicted Digit: {predicted_digit}"
            )

            st.write(
                f"Confidence: {confidence:.2f}%"
            )

            st.subheader("Prediction Probabilities")

            for digit, probability in enumerate(
                prediction[0]
            ):
                st.write(
                    f"{digit}: {probability * 100:.2f}%"
                )

            st.subheader("Processed Image")

            st.image(
                img_array.reshape(28, 28),
                width=250,
                clamp=True
            )

            st.write(
                "Minimum pixel:",
                float(img_array.min())
            )

            st.write(
                "Maximum pixel:",
                float(img_array.max())
            )

            st.write(
                "Non-zero pixels:",
                int(np.count_nonzero(img_array))
            )

st.caption(
    "Deep Learning Mini Project | ANN | MNIST"
)
