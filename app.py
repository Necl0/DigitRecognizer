from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pickle
import random

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
    background_color= "White",
    drawing_mode=drawing_mode,
    key="canvas",
    stroke_color="rgba(0, 0, 0, 1)",
)
def gen_problem():
    kind = random.choice(['add','sub','mul','div'])
    while True:
        a,b = [random.randrange(-99,99) for _ in range(2)]
        match kind:
            case 'add':
                if a+b in range(10):
                    return f"{a} + {b} = {a+b}"
            case 'sub':
                if a-b in range(10):
                    return f"{a} - {b} = {a-b}"
            case 'div':
                if a//b in range(10):
                    return f"{a} / {b} = {a//b}"
            case 'mul':
                if a*b in range(10):
                    return f"{a} * {b} = {a*b}"

st.write("Equation: ")
st.write([gen_problem() for _ in range(1)])


if st.button("Predict"):
    model = pickle.load(open('model.pkl', 'rb'))

    # convert canvas content to png
    img = Image.fromarray(np.uint8(canvas_result.image_data))


    img.save('temp.png')

    # convert image to numpy array
    img = 1 - (np.asarray(Image.open("./temp.png").convert("L").resize((28, 28))) / 255)

    prediction = model.predict(img[None, :, :]).argmax()
    print(prediction)
    # display result
    st.write("Prediction: ", prediction)
