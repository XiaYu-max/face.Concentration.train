
import pandas as pd
import torch
import numpy as np
import cv2
import torch.utils.data as data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class FaceDataset(data.Dataset):

    def __init__(self, root):
        super(FaceDataset, self).__init__()
        self.root = root

        df_path = pd.read_csv(root + '\\dataset.csv', header = None, usecols = [0])
        df_label = pd.read_csv(root + '\\dataset.csv', header = None, usecols = [1])

        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]

    def __getitem__(self, item):
        face = cv2.imread(self.root + '\\' + self.path[item])

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        face_Gus = cv2.GaussianBlur(face_gray, (3,3), 0)

        face_hist = cv2.equalizeHist(face_gray)

        face_normalized = face_hist.reshape(1, 48, 48)
        
        face_tensor = torch.from_numpy(face_normalized)
        face_tensor = face_tensor.type('torch.FloatTensor')
        face_tensor = face_tensor.to(device)
        label = self.label[item]
        return face_tensor, label

    def __len__(self):
        return self.path.shape[0]