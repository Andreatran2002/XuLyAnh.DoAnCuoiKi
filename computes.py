import cv2
import numpy as np

from PIL import Image


def run(type, upload_file, c):
    if (type == 'Negative'):
        return negative(upload_file)
    elif (type == 'Logarit'):
        return logarit(upload_file, c)
    elif (type == 'Picecewise Linear'):
        return piecewise(upload_file)
    elif (type == 'Histogram'):
        return histogram(upload_file)
    return []


def negative(upload_file):
    image = Image.open(upload_file)

    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    negative_img = 255 - img
    return negative_img


def log_transform(img, c):
    log_img = c * np.log10(1 + img)
    log_img = np.uint8(log_img)
    return log_img


def logarit(uploaded_file, c):
    pil_img = Image.open(uploaded_file)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Define the constant for the logarithmic transformation
    c = 255 / np.log10(1 + np.max(img))

    # Apply the logarithmic transformation to the image
    log_img = log_transform(img, c)
    return log_img


def piecewise_linear(img, breakpoints, slopes, intercepts):
    piecewise_img = np.piecewise(img, [img < breakpoints[0], (img >= breakpoints[0]) & (img < breakpoints[1]), img >= breakpoints[1]], [
                                 lambda x: slopes[0] * x + intercepts[0], lambda x: slopes[1] * x + intercepts[1], lambda x: slopes[2] * x + intercepts[2]])
    return piecewise_img


def piecewise(uploaded_file):
    pil_img = Image.open(uploaded_file)
    img = np.asarray(pil_img.convert('L'))

    # Define the breakpoints for the piecewise linear function
    breakpoints = [0, 127, 255]

    # Define the slope and intercept for each segment of the piecewise linear function
    slopes = [1, 2, 1]
    intercepts = [0, -127, 127]

    # Apply the piecewise linear function to the image
    piecewise_img = piecewise_linear(img, breakpoints, slopes, intercepts)
    return piecewise_img


def histogram_linear(img):
    # Chuyển ảnh sang ảnh xám
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Tính histogram của ảnh
    hist, bins = np.histogram(gray_img, 256, [0, 256])

    # Tính tổng số pixel trong ảnh
    total_pixels = gray_img.shape[0] * gray_img.shape[1]

    # Tính tỉ lệ giới hạn dưới và giới hạn trên
    cdf = hist.cumsum()
    cdf_normalized = cdf / total_pixels
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Áp dụng phép biến đổi tuyến tính để tăng cường độ tương phản của ảnh
    gray_img = cdf[gray_img]

    # Chuyển ảnh xám về ảnh màu
    img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    return img


def histogram(uploaded_file):
    img = Image.open(uploaded_file)
    img = np.array(img)

    # Áp dụng phương pháp Histogram Linear để tăng cường độ tương phản của ảnh
    enhanced_img = histogram_linear(img)
    return enhanced_img
