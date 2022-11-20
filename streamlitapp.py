from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pickle

# title of the app
st.title("Digit Recognizer")

# Specify canvas parameters in application
drawing_mode = "freedraw"
stroke_width = 3
realtime_update = True

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    update_streamlit=realtime_update,
    height=100,
    width=100,
    drawing_mode=drawing_mode,
    key="canvas",
)

def predict_img(img):
    model = pickle.load(open('model.pkl', 'rb'))

    # predict digit
    prediction = model.predict(img)
    return prediction

# add button that when clicked calls predict_img on numpy array version of canvas content
if st.button("Predict"):
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGB')
    img = img.convert('L')
    # preprocess image
    img = np.array(img.resize((28, 28))).reshape(1, 784)
    prediction = predict_img(img)[0]
    st.write(f"Predicted Digit: {prediction}")

