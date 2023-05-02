import cv2
import numpy as np
import streamlit as st
from PIL import Image

from computes import run

st.set_page_config(page_title="Xử lý ảnh", page_icon="🖼️")

st.markdown("# Xử lý ảnh")
st.sidebar.header("Xử lý ảnh")

# IMREAD_options = ['GRAYSCALE', 'COLOR']

# selected_IMREAD = st.sidebar.selectbox('Group', IMREAD_options)

options = {
    'Chapter 3': [
        'Negative',
        'Logarit',
        'Picecewise Linear',
        'Histogram',
        "HistEqual",
        "HistEqualColor",
        "LocalHist",
        "HistStat",
        "BoxFilter",
        "LowpassGauss",
        "Threshold",
        "MedianFilter",
        "Sharpen",
        "Gradient",
    ],
    'Chapter 4': [
        "Spectrum",
        "FrequencyFilter",
        "DrawNotchRejectFilter",
        "RemoveMoire"
    ],
    'Chapter 5': [
        "CreateMotionNoise",
        "DenoiseMotion",
        "DenoisestMotion",
    ],
    'Chapter 9': [
        "Erosion",
        "Dilation",
        "OpeningClosing",
        "Boundary",
        "HoleFilling",
        "HoleFillingMouse",
        "ConnectedComponent",
        "CountRice",
    ]
}

selected_option = st.sidebar.selectbox('Group', list(options.keys()))

selected_type = st.sidebar.selectbox('Group', options[selected_option])


# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])


beforeCol, afterCol = st.columns(2)

# Display negative image
if uploaded_file is not None:
    c = 0
    img = Image.open(uploaded_file)
    if (selected_type == 'Logarit'):
        c = st.slider("Hệ số c", min_value=1, max_value=100, value=50, step=1)

    # Display original and negative images
    with beforeCol:
        st.subheader('Original image')
        st.image(img, channels='BGR', use_column_width=True)

    if st.button('Run magic 🎆', key='run_btn'):
        rs_img = run(selected_type, uploaded_file, c)

        if (len(rs_img) != 0):
            with afterCol:
                st.subheader(f'${selected_type} image')
                st.image(rs_img, use_column_width=True)
