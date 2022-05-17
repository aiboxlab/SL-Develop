import csv


import cv2
import dlib
import pandas as pd
from imutils import face_utils

p = "../../../models/shape_predictor_68_face_landmarks.dat"

image_name = 'img_video_frame_1.png'
image = cv2.imread("../../../data/examples/"+image_name+".jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

rects = detector(gray, 0)
print("----------------------")
print(rects)
print("----------------------")

df = pd.DataFrame(columns=["landmark", "x", "y"])


for (i, rect) in enumerate(rects):
    print(i)
    print(rect)

    if rects != False:

        # Get the shape using the predictor
        landmarks = predictor(gray, rect)
        shape = face_utils.shape_to_np(landmarks)

        # Defining x and y coordinates of a specific point
        # x=landmarks.part(48).x
        # y=landmarks.part(48).y
        # Drawing a circle

        w = 20
        h = 2

        # Start coordinate, here (100, 50)
        # represents the top left corner of rectangle
        start_point = (100, 50)

        # Ending coordinate, here (125, 80)
        # represents the bottom right corner of rectangle
        end_point = (150, 80)  # (largura, altura)

        # Black color in BGR
        color = (0, 0, 0)

        # Line thickness of -1 px
        # Thickness of -1 will fill the entire shape
        thickness = -1

        image = cv2.rectangle(image, start_point, end_point, color, thickness)

        # for (x, y) in shape:
        # print(x,y, i)
        # cv2.circle(image, (x, y), 4, (0, 0, 255), -1)

        # Draw black background rectangle
        # cv2.rectangle(image, (x, x), (x+20, y+ 1), (255,255,255), -1)

        for n in range(0, 68):
            mark = n
            x = landmarks.part(mark).x
            y = landmarks.part(mark).y

            overlay = image.copy()

            cv2.circle(overlay, (x, y), 10, (255, 255, 255), -1)

            alpha = 0.40
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            """cv2.putText(
                image,
                text=str(mark + 1),
                org=(x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.3,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )"""

            new_row = {
                    "landmark": n + 1,
                    "x": x,
                    "y": y,
                }
            df = df.append(new_row, ignore_index=True)


cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.imwrite("../../../data/examples/"+image_name+"_lands_dlib.jpg", image)
df.to_csv(
        "../../../data/examples/{0}_landmaks_dlib" + str(".csv").format(image_name), sep=";"
    )
