import cv2

from ultralytics import YOLO

# 載入模型
model = YOLO("/home/arthur/ultralytics/runs/detect/alphabet_200_model(m)/weights/best.pt")

cap = cv2.VideoCapture(0)

print("辨識啟動中... (僅顯示類別 ID)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
    results = model(frame, conf=0.4, device="0", verbose=False)

    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        # 將資料轉到 CPU
        classes = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for cls_id, conf in zip(classes, confs):
            # 直接印出數字 ID
            print(f"ID: {int(cls_id)} | Conf: {conf:.2f}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
