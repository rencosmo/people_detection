import numpy as np
import cv2
import time

if __name__ == "__main__":
    cap = cv2.VideoCapture("rtsp://admin:RRRQGG@192.168.1.11:554/h264/ch1/main/av_stream")
    # cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        try:
            r, img = cap.read()
        except:
            r = False

        if r is True:
            img = cv2.resize(img, (1280, 960))
            cv2.imshow("preview", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
