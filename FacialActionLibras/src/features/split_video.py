import cv2
import os
import numpy as np
import pandas as pd
from imutils import face_utils


def create_directory(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")
    return dirName


video_output = "../../data/processed/examples/split/"
create_directory(video_output)
video_input = "../../data/examples/v_PlayingFlute_g03_c07.avi-mediapipe.avi"
video_name = video_input.split("/")[-1]

cap = cv2.VideoCapture(video_input)
last_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(last_frame_num)
listas = np.array_split(range(last_frame_num), 32)
print(listas)

tuplas = []
for i in listas:
    # print(min(i), max(i))
    tuplas.append((min(i), max(i)))
# Converta a lista em tuplas (max e min)
print(tuplas)
# parts = [(1, 2), (2, 3)]

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# size = (frame_width, frame_height)
# result = cv2.VideoWriter(video_output+'v_ShavingBeard_g03_c03-dlib.avi',
# cv2.VideoWriter_fourcc(*'MJPG'),
# 25, size)

ret, frame = cap.read()
h, w, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc(*"XVID")

writers = []
for start, end in tuplas:
    writers.append(
        cv2.VideoWriter(
            "{}part{}-{}.avi".format(video_output, start, end),
            fourcc,
            20.0,
            (w, h),
        )
    )
# writers = [cv2.VideoWriter(f"part{start}-{end}.avi", fourcc, 20.0, (w, h)) for start, end in parts]

f = 0
while ret:
    f += 1
    for i, part in enumerate(tuplas):
        start, end = part
        if start <= f <= end:
            writers[i].write(frame)
    ret, frame = cap.read()

for writer in writers:
    writer.release()

cap.release()
