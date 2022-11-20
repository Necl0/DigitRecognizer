from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pickle
from random import randint

# title of the app
st.title("Digit Recognizer")

def getop(op):
    ans = -1
    while ans not in range(10):
        a = randint(-99,99)
        b = randint(-99,99)
        if op == '+': ans = a+b
        if op == '*': ans = a*b
    return f'{a:3} {op} {b:3}'

st.write("Equation: ")
eq = getop('+')

st.write(eq)
is_done = True

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
    # if prediction == ans:
    #     st.write("Correct!")
    # else:
    #     st.write("Incorrect!")
        
