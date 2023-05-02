import argparse

import numpy as np
import cv2 as cv
from unidecode import unidecode
import streamlit as st
import sys
import os

sys.path.insert(0, './code/FaceDetection')


if 'stop' not in st.session_state:
    st.session_state.stop = True
    stop = True


def convert_name(name):
    name = unidecode(name)
    name = name.lower()
    name = name.replace(' ', '_')
    return name


st.set_page_config(page_title="Training Model", page_icon="📈")

st.markdown("# Thêm dữ liệu mới ")

name = st.text_input('Nhận tên', '')
st.write('Dữ liệu sẽ được lưu lại trong folder : models/'+convert_name(name))
getModelBtn = st.button('Bắt đầu/Kết thúc training')

if getModelBtn == True:
    if name == "":
        st.warning('Vui lòng hãy nhập tên đối tượng', icon="⚠️")


FRAME_WINDOW = st.image([])
deviceId = 0
cap = cv.VideoCapture(deviceId)
for camera_idx in range(15):
    cap = cv.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"Camera {camera_idx} is not available")
    else:
        print(f"Camera {camera_idx} is available")
        # Thực hiện các thao tác với camera ở đây


def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError


parser = argparse.ArgumentParser()
# parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. Omit for detecting on default camera.')
# parser.add_argument('--image2', '-i2', type=str, help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
# parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=1.0,
                    help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='./data/face_detection_yunet_2022mar.onnx',
                    help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='./data/face_recognition_sface_2021dec.onnx',
                    help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold', type=float, default=0.9,
                    help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3,
                    help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000,
                    help='Keep top_k bounding boxes before NMS.')
# parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
args = parser.parse_args()


if getModelBtn:
    if st.session_state.stop == False:
        st.session_state.stop = True
        cap.release()
    else:
        st.session_state.stop = False


print('Trang thai nhan Stop', st.session_state.stop)

if 'frame_stop' not in st.session_state:
    frame_stop = cv.imread('./asset/stop.jpg')
    st.session_state.frame_stop = frame_stop
    print('Đã load stop.jpg')

trainingBtn = st.button('Training cho mẫu')

if st.session_state.stop == True:
    FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')


def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(
                idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0] +
                         coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]),
                      2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]),
                      2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


detector = cv.FaceDetectorYN.create(
    args.face_detection_model,
    "",
    (320, 320),
    args.score_threshold,
    args.nms_threshold,
    args.top_k
)
recognizer = cv.FaceRecognizerSF.create(args.face_recognition_model, "")


tm = cv.TickMeter()
cap = cv.VideoCapture(0)
frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
detector.setInputSize([frameWidth, frameHeight])
dem = 0


while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        print('No frames grabbed!')
        break
    # Inference
    tm.start()
    faces = detector.detect(frame)  # faces is a tuple
    tm.stop()

    visualize(frame, faces, tm.getFPS())

    if faces[1] is not None and name != "":
        if os.path.exists('./model/images/'+convert_name(name)) == False:
            os.mkdir('./model/images/'+convert_name(name))
        print("đang chụp")
        face_align = recognizer.alignCrop(frame, faces[1][0])
        file_name = './model/images/' + \
            convert_name(name)+'/'+convert_name(name)+'_%04d.bmp' % dem
        cv.imwrite(file_name, face_align)
        dem = dem + 1

    # print(key)
    # Draw results on the input image

    if st.session_state.stop == True:
        break

    # Visualize results
    FRAME_WINDOW.image(frame, channels='BGR')

cv.destroyAllWindows()
