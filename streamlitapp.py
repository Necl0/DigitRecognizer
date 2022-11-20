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

# add button that when clicked calls predict_img on numpy array version of canvas content
if st.button("Predict"):
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGB')
    img = img.convert('L')
    # preprocess image
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img /= 255
    # predict digit
    prediction = predict_img(img)
    print(prediction)
    st.write("The digit is: ", np.argmax(prediction))

