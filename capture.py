from xmlrpc.client import NOT_WELLFORMED_ERROR
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import dlib
import time
import imutils
from imutils.face_utils import rect_to_bb
from imutils.video import WebcamVideoStream
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

now = 0

model_name = "mmod_human_face_detector.dat"
detector = dlib.cnn_face_detection_model_v1(model_name)

# detector2 = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('dlib/shape_predictor_68_face_landmarks.dat')
# pred = np.argmax(pred.data.cpu().numpy(), axis = 1)
font = cv2.FONT_HERSHEY_SIMPLEX

vs = WebcamVideoStream().start()
i = 6067
flag = 0
while True:
    # 取得當前的frame，轉成RGB圖片
    frame = vs.read()
    img = frame.copy()
    img = imutils.resize(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(rgb, 0)
    ratio = frame.shape[1] / img.shape[1]
   
    # faces2 = detector2(rgb, 0)

    # if len(faces2) != 0:
    # # 检测到人脸
    #     for i in range(len(faces2)):
    #         # 取特征点坐标
    #         landmarks = np.matrix([[p.x, p.y] for p in predictor(rgb, faces2[i]).parts()])
    #         for idx, point in enumerate(landmarks):
    #             # 68 点的坐标
    #             pos = (point[0, 0], point[0, 1])

    #             # 利用 cv2.circle 给每个特征点画一个圈，共 68 个
    #             cv2.circle(frame, pos, 2, color=(139, 128 , 120))
    #             # 利用 cv2.putText 写数字 1-68


    results = detector(rgb, 0)
    boxes = [rect_to_bb(r.rect) for r in results]
    for box in boxes:
            # 計算bounding box(邊界框)與準確率 - 取得(左上X，左上Y，右下X，右下Y)的值 (記得轉換回原始frame的大小)
        box = np.array(box) * ratio
        (x, y, w, h) = box.astype("int")
        crop = img[y:y+h, x:x+w]
        
        # 畫出邊界框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        time.sleep(0.07)
    
    get_time = datetime.now()
    second = get_time.second

    if second - now == 1:
        img = cv2.resize(crop, (48, 48))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        path = 'C:\\face_detection\data\\face\\total\\' + str(i) + '.jpg'
        cv2.imwrite(path, img)
        i += 1
        now = second
        print(i)
    elif flag == 0:
        now = second
        flag = 1
    elif second == 0:
        now = second

    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        cv2.destroyAllWindows()
        break

