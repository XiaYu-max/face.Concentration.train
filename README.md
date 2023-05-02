# 用於人臉專注度的訓練模型
## 摘要
  本篇為訓練人臉表情是否上課專心模型的研究，此連結只有訓練模型，另有一篇是為寫成RPC連線的專案用於介面呈現。
## 實做軟體
  * Anaconda
  ```Python
    python 3.6
    pip install cmake
    pip install dlib # 此為cpu版，gpu版參考官方或網路上資源
    pip install pandas
    pip install opencv-python
  ```
  安裝torch(請參照torch官方下載)，安裝cuda(參照自己顯卡版本安裝cuda的驅動)，研究以cuda 11.7版本。
  上述為較重要的py套件，如有遺漏py套件，可直接查詢安裝。
## 檔案說明 (部分檔案沒上傳，ex: data檔案大太，因此沒上傳;詳情可見.gitnore)
  * main.py 
    main.py是主要執行資料預處理，訓練、預測、攝像頭即時檢測，參數可參考程式碼
  * DataProcess.py & Dataset.py
    研究參考fer2013的資料集與code，所以如果檔案為fer2013的train.csv&val.csv，可利用main執行程式。
  * train.py
    主要用於訓練的檔案，模型參數內容可見faceCNN.py
  * faceCNN.py
    建立模型大小、層數的宣告，之後介面呈現也有用此py檔
  * visualize.py
    可簡單預測單張照片或開啟攝像頭檢測人臉與情緒
  * capture.py
    此檔案為建立資料的程式，可即時錄影並裁剪人臉為48*48的照片並以cv.write寫成jpg檔，用於後續訓練。
