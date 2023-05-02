import argparse

import numpy as np
import cv2 as cv
from unidecode import unidecode
import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib

from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.manifold import TSNE


if 'stop' not in st.session_state:
    st.session_state.stop = True
    stop = True

if 'trainingMessage' not in st.session_state:
    st.session_state.trainingMessage =  ""
def convert_name(name):
    name = unidecode(name)
    name = name.lower()
    name = name.replace(' ', '_')
    return name


st.set_page_config(page_title="Training Model", page_icon="📈")

st.markdown("# Thêm dữ liệu mới ")

name = st.text_input('Nhập tên', '')
st.write('Dữ liệu sẽ được lưu lại trong folder : models/'+convert_name(name))
getModelBtn = st.button('Bắt đầu/Kết thúc training')

if getModelBtn == True:
    if name == "":
        st.warning('Vui lòng hãy nhập tên đối tượng', icon="⚠️")


FRAME_WINDOW = st.image([])
deviceId = 0
for camera_idx in range(15):
    cap = cv.VideoCapture(camera_idx)
    if not cap.isOpened():
        deviceId = camera_idx
        print(f"Camera {camera_idx} is not available")
        break
    else:
        print(f"Camera {camera_idx} is available")
        # Thực hiện các thao tác với camera ở đây
cap = cv.VideoCapture(deviceId)


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



if 'frame_stop' not in st.session_state:
    frame_stop = cv.imread('./asset/stop.jpg')
    st.session_state.frame_stop = frame_stop


class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 
    
def load_metadata(path):
    metadata = []
    for i in sorted(os.listdir(path)):
        for f in sorted(os.listdir(os.path.join(path, i))):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg' or ext == '.bmp':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

def load_image(path):
    img = cv.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

def align_image(img):
    pass

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def show_pair(idx1, idx2):
    plt.figure(figsize=(8,3))
    plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()))

def TrainingModel():
    detector = cv.FaceDetectorYN.create(
        "./data/face_detection_yunet_2022mar.onnx",
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )
    detector.setInputSize((320, 320))

    recognizer = cv.FaceRecognizerSF.create(
                "./data/face_recognition_sface_2021dec.onnx","")


    metadata = load_metadata('./model/images')

    embedded = np.zeros((metadata.shape[0], 128))

    for i, m in enumerate(metadata):
        st.session_state.trainingMessage += "\n" +  m.image_path()
        img = cv.imread(m.image_path(), cv.IMREAD_COLOR)
        face_feature = recognizer.feature(img)
        embedded[i] = face_feature

    targets = np.array([m.name for m in metadata])

    encoder = LabelEncoder()
    encoder.fit(targets)

    # Numerical encoding of identities
    y = encoder.transform(targets)

    train_idx = np.arange(metadata.shape[0]) % 5 != 0
    test_idx = np.arange(metadata.shape[0]) % 5 == 0
    X_train = embedded[train_idx]
    X_test = embedded[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    svc = LinearSVC()
    svc.fit(X_train, y_train)
    acc_svc = accuracy_score(y_test, svc.predict(X_test))
    st.session_state.trainingMessage += "\n" + 'SVM accuracy: %.6f' % acc_svc
    joblib.dump(svc,'./model/output/svc.pkl')


def onClickTraining():
    TrainingModel()

st.markdown("# Training cho mẫu ")
trainingBtn = st.button('Training cho mẫu', on_click=onClickTraining)
st.text_area("Training Message" ,value=st.session_state.trainingMessage, height=200)


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
        break
    if st.session_state.stop == True:
        break
    # Inference
    tm.start()
    faces = detector.detect(frame)  # faces is a tuple
    tm.stop()

    visualize(frame, faces, tm.getFPS())

    if faces[1] is not None and name != "":
        if os.path.exists('./model/images/'+convert_name(name)) == False:
            os.mkdir('./model/images/'+convert_name(name))
        face_align = recognizer.alignCrop(frame, faces[1][0])
        file_name = './model/images/' + \
            convert_name(name)+'/'+convert_name(name)+'_%04d.bmp' % dem
        cv.imwrite(file_name, face_align)
        dem = dem + 1

    # print(key)
    # Draw results on the input image

   
    # Visualize results
    FRAME_WINDOW.image(frame, channels='BGR')

cv.destroyAllWindows()
