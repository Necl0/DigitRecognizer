from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pickle
import cv2

# title of the app
st.title("Digit Recognizer")

# Specify canvas parameters in application
drawing_mode = "freedraw"
stroke_width = 3
realtime_update = True

# Create a canvas component
canvas_result = st_canvas(
    stroke_width=stroke_width,
    update_streamlit=realtime_update,
    height=100,
    width=100,
    drawing_mode=drawing_mode,
    key="canvas",
    stroke_color="rgba(0, 0, 0, 1)",
)


def predict_img(img):
    model = pickle.load(open('model.pkl', 'rb'))

    # predict digit
    prediction = model.predict(img)
    return prediction


if st.button("Predict"):

    # convert canvas content to png
    img = Image.fromarray(np.uint8(canvas_result.image_data))
    img.save('temp.png')

    # read image
    img = Image.open('temp.png')

    # convert image to numpy array
    img = np.array(img)

    # resize image to 28x28
    img = cv2.resize(img, (14, 14))

    # reshape image
    img = img.reshape(1, 28, 28, 1)

    # predict digit
    prediction = predict_img(img)


    # display result
    st.write("Prediction: ", prediction.argmax())

    st.write("The probability is: ", prediction.max())
