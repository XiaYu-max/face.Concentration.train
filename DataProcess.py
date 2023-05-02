import pandas as pd
import numpy as np
import cv2
import os, sys, time

from sqlalchemy import false
def DataProcess():
    
    path = '.\\data\\train.csv'
    df = pd.read_csv(path)
    df = df.fillna(0)
    
    df_y = df[['emotion']]
    df_x = df[['pixels']]
    if os.path.exists('.\\data\\label.csv') and os.path.exists('.\\data\\data.csv'):
        df_y.to_csv('.\\data\\label.csv', index = False, header = False)
        df_x.to_csv('.\\data\\data.csv', index = False, header = False)

    path = '.\\data\\face'
    data = np.loadtxt('.\\data\\data.csv')
    
    for i in range(data.shape[0]):
        k = i + 1
        percentage = 100 * float(k)/float(len(data))
        str = '>'*(int(percentage)//2) + ' '*((100-int(percentage))//2)
        sys.stdout.write('\r' + str + '[%s%%]'%(int(percentage)))
        face_array = data[i, :].reshape((48, 48))
        cv2.imwrite(path + '//' + '{0}.jpg'.format(i), face_array)
    print("處理完畢")


def data_label(path):

    df_label = pd.read_csv('.\\face_detection\\data\\label.csv', header = None)

    files_dir = os.listdir(path)

    path_list = []
    label_list = []

    for files_dir in files_dir:
        if os.path.splitext(files_dir)[1] == ".jpg":
            path_list.append(files_dir)
            index = int(os.path.splitext(files_dir)[0])
            label_list.append(df_label.iat[index, 0])

    path_s = pd.Series(path_list)
    label_s = pd.Series(label_list)
    df = pd.DataFrame()
    df['path'] = path_s
    df['label'] = label_s
    df.to_csv(path + '\\dataset.csv', index = False, header = False)
