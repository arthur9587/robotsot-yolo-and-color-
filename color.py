import sys

import cv2
import numpy as np

# 開啟相機
webcam = cv2.VideoCapture(0)

# 如果無法開啟相機，則退出
if not webcam.isOpened():
    print("錯誤：無法開啟相機。")
    sys.exit()

while True:
    # 讀取相機畫面
    success, imageFrame = webcam.read()
    if not success:
        print("錯誤：無法讀取畫面。")
        break

    # 將 BGR 轉換為 HSV 色彩空間
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # --- 1. 定義顏色範圍 (HSV) --

    # # 紅
    # red_lower = np.array([136, 87, 111], np.uint8)
    # red_upper = np.array([180, 255, 255], np.uint8)

    # 綠色
    green_lower = np.array([35, 52, 72], np.uint8)  # Hue 從 25 調到 35
    green_upper = np.array([102, 255, 255], np.uint8)

    # 藍色
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)

    # --- 加入黃色範圍 ---
    yellow_lower = np.array([15, 100, 100], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)

    #  2. 製作掩膜 (Masks) --
    # red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    # --- 3. 形態學處理 (去除雜訊) ---
    kernel = np.ones((5, 5), "uint8")

    # 使用 dilate (膨脹) 讓顏色區域更完整，減少雜點
    # red_mask = cv2.dilate(red_mask, kernel)
    green_mask = cv2.dilate(green_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)
    yellow_mask = cv2.dilate(yellow_mask, kernel)

    # --- 4. 追蹤與標記顏色的函式 --
    def detect_and_draw(mask, color_text, bgr_color):
        # 尋找輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 計算輪廓面積
            area = cv2.contourArea(contour)

            # 只有面積大於一定大小的才進行標記，避免環境雜訊
            if area > 2000:  # 稍微提高面積門檻，讓辨識更乾淨
                # 取得包絡矩形的座標與寬高
                x, y, w, h = cv2.boundingRect(contour)

                # 1. 畫出矩形框 (在原圖上)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), bgr_color, 2)

                # 2. 寫上顏色文字 (中文文字需要特殊處理，這裡使用英文以保持代碼簡潔)
                cv2.putText(imageFrame, color_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr_color, 2)

    # --- 5. 執行辨識 (呼叫函式) ---

    # detect_and_draw(red_mask, "Red", (0, 0, 255))
    detect_and_draw(green_mask, "Green", (0, 255, 0))
    detect_and_draw(blue_mask, "Blue", (255, 0, 0))

    # 黃色的 BGR 是 (0, 255, 255)
    detect_and_draw(yellow_mask, "Yellow", (0, 255, 255))

    # --- 6. 顯示結果與結束 ---
    cv2.imshow("Multi-Color Detection (R,G,B,Y)", imageFrame)

    # 如果需要同時看 Mask，可以解鎖下面這行 (例如看黃色的 Mask)
    # cv2.imshow("Yellow Mask", yellow_mask)

    # 按下 'q' 鍵退出程式
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

# 釋放相機與關閉視窗
webcam.release()
cv2.destroyAllWindows()
