import cv2
import numpy as np
from ultralytics import YOLO

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import pyzed.sl as sl
import ogl_viewer.tracking_viewer as gl
import math
import os
from threading import Lock, Thread
from time import sleep
from time import time
import argparse

track_history = defaultdict(lambda: [])
model = YOLO("best.pt")
names = model.model.names

frame = cv2.imread("img4.png")
x1, y1 = 975, 390
x2, y2 = 1040, 455

# Görüntüyü crop edin
cropped_image = frame[y1:y2, x1:x2]

dim=(640,640)
            
imgz = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_CUBIC) #INTER_CUBIC interpolation

#w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
# Sol üst ve sağ alt köşelerin koordinatları
# x1, y1 = 1030, 560
# x2, y2 = 1064, 600

# # Görüntüyü crop edin
# cropped_image = frame[y1:y2, x1:x2]

# dim=(320,320)
            
# imgz = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_CUBIC) #INTER_CUBIC interpolation


results = model.predict(frame,save=True)
boxes = results[0].boxes.xyxy.tolist()

print(boxes)



