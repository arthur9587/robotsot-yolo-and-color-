import os

import cv2

from ultralytics import YOLO

# 1. 載入模型
model = YOLO("/home/arthur/ultralytics/runs/detect/color_200_model/weights/best.pt")

# 2. 設定你要測試的圖片路徑
img_path = "/home/arthur/Downloads/test.jpg"  # 請確保檔案在同一個資料夾，或使用絕對路徑

if not os.path.exists(img_path):
    print(f"錯誤：找不到圖片檔案 {img_path}")
else:
    # 3. 執行推論
    # imgsz=640 確保與訓練時的解析度接近
    results = model(img_path, conf=0.25, device="cuda:0")

    # 4. 取得結果並顯示
    res = results[0]
    annotated_img = res.plot()

    # 5. 印出詳細結果
    print("\n--- 辨識結果統計 ---")
    if len(res.boxes) == 0:
        print("未偵測到任何物體。")
    else:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = res.names[cls_id]
            print(f"標籤: {name:10} | 信心度: {conf:.4f}")

    # 6. 顯示圖片視窗
    cv2.imshow("YOLOv8 Image Test", annotated_img)
    print("\n按任意鍵關閉視窗...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
