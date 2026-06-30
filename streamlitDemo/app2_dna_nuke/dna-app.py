import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image, ImageSequence

# image
img = Image.open("image.png")
imwidth, imheight = img.size
img=img.crop((0,250,imwidth,imheight))
st.image(img, caption="Cropped Banana Cat", use_container_width=True)

# GIF 

# Display the original GIF file to preserve animation
st.image("spinning-cat.gif", caption="Spinning Cat (Animated)", use_container_width=True)
