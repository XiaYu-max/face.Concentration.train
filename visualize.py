from sqlalchemy import null
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from FaceCNN import *
import Dataset
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import cv2
import dlib
import time
import imutils
from imutils.face_utils import rect_to_bb
from imutils.video import WebcamVideoStream
from xmlrpc.client import ServerProxy

#classes = ('Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral')
classes = ('focus', 'unfous')
#transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
def load_img(face):
    #face_Gus = cv2.GaussianBlur(face_gray, (3,3), 0)

    face_hist = cv2.equalizeHist(face)

    face_normalized = face_hist.reshape(1, 48, 48)

    face_tensor = torch.from_numpy(face_normalized)
    face_tensor = face_tensor.type('torch.FloatTensor')
    img = torch.autograd.Variable(face_tensor,requires_grad = True)
    img = img.unsqueeze(0)
    return img.to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def visualize(model):
    crop, show, last = 0, 0, ''
    model_name = "mmod_human_face_detector.dat"
    detector = dlib.cnn_face_detection_model_v1(model_name)
    # pred = np.argmax(pred.data.cpu().numpy(), axis = 1)
    vs = WebcamVideoStream().start()
    start = time.time()

    fps = vs.stream.get(cv2.CAP_PROP_FPS)

    prediction = "沒有結果"
    while True:
        # 取得當前的frame，轉成RGB圖片
        frame = vs.read()
        img = frame.copy()
        img = imutils.resize(img, width=500)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ratio = frame.shape[1] / img.shape[1]

        results = detector(rgb, 0)
        boxes = [rect_to_bb(r.rect) for r in results]
       
        for box in boxes:
            # 計算bounding box(邊界框)與準確率 - 取得(左上X，左上Y，右下X，右下Y)的值 (記得轉換回原始frame的大小)
            box = np.array(box) * ratio
            (x, y, w, h) = box.astype("int")
            crop = frame[y:y+h, x:x+w]
            
            # 畫出邊界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            time.sleep(0.07)

        image = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(image, (48, 48))  
        img = load_img(img)
        out = model.forward(img)
        _, predicted = torch.max(out, 1)
        prediction = classes[predicted[0]]

        end = time.time()
        global num1, num2, num3, num4
        if (end - start) != 0 and  abs(end - show) > 1:
            cv2.putText(frame, f"FPS: {str(int(1 / (end - start)))} {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            show = time.time()
            num1, num2, num3, num4 = frame, end, start, prediction
        else :
            cv2.putText(frame, f"FPS: {str(int(1 / (num2 - num3)))} {num4}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        
        # 顯示影像
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        if key == ord('c'):
            s = ServerProxy('http://localhost:6666', allow_none = True)
            s.set('資料', prediction)
            last = key
        if key == ord('s'):
            s.set('資料', "no_result")
            last = key

        if last == ord('c') and abs(end - show) > 3:
            s.set('資料', prediction)
        start = end

