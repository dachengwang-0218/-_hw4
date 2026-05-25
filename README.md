# AIoT HW4 專案說明

本專案為物聯網課程 HW4 作業內容，主要包含手勢辨識相關的圖片、展示影片、模型訓練程式與模型比較結果。專案聚焦於石頭、布、剪刀三分類任務，並比較不同模型方法的效果。

## 專案架構

```text
.
├── AI協作對話.md
├── README.md
├── part1/
│   ├── part1_camera.jpg
│   └── part1_test.jpg
├── part2/
│   └── part2_demo.mp4
├── part3/
│   ├── part3_mlp_result.png
│   ├── part3_mobilenetv3_result.png
│   └── part3_更換模型原因與差異比較.md
└── 程式碼/
    ├── compare_three_models.py
    ├── train_mlp.py
    ├── train_mobilenetv3.py
    └── train_svm.py
```

## 檔案與資料夾說明

### `AI協作對話.md`

記錄本次作業過程中與 AI 協作的對話內容，可用於說明開發、除錯與模型調整的歷程。

### `part1/`

存放第一部分作業使用或產出的圖片。

- `part1_camera.jpg`：camera.py執行結果。
- `part1_test.jpg`：test.py執行結果。

### `part2/`

存放第二部分作業的展示影片。

- `part2_demo.mp4`：10個手勢的demo影片。

### `part3/`

存放第三部分的模型實驗結果與分析文件。

- `part3_mlp_result.png`：MLP 模型的評估結果。
- `part3_mobilenetv3_result.png`：MobileNetV3 模型的評估結果。
- `part3_更換模型原因與差異比較.md`：說明更換模型的原因，以及不同模型之間的差異比較。

### `程式碼/`

存放模型訓練與比較用的 Python 程式。

- `train_svm.py`：將圖片轉為灰階並縮放為 64x64，再攤平成一維特徵以訓練 SVM 分類器。
- `train_mlp.py`：使用 MediaPipe HandLandmarker 擷取手部 21 個 landmark，並訓練 MLP 分類器。
- `train_mobilenetv3.py`：使用 MobileNetV3-Small 進行影像分類模型訓練。
- `compare_three_models.py`：比較 SVM、MLP 與 MobileNetV3 三種模型在測試資料上的表現。
