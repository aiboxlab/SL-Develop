import cv2
import numpy as np
import mediapipe as mp
import os
import dlib
import pandas as pd
from imutils import face_utils
#from protobuf_to_dict import protobuf_to_dict

# Importação da imagem e arquivo de mapeamento
image_name = 'annotated_image_detector'
lands_mp_match_dlib_682d = pd.read_csv('../../data/examples/lands_mp_match_dlib_682d_ref02.csv', sep=';')
image = cv2.imread("../../data/examples/"+image_name+".png")
image_origin = image.copy()

# Configurações do Dlib
p = "../../models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 0)  # delimitação do quadro que contém a face

##### Execução do Dlib

for (i, rect) in enumerate(rects):
    print(i, rects, rects[0], rect)
    if rects != False:
        # Get the shape using the predictor
        landmarks = predictor(gray, rect)
        shape = face_utils.shape_to_np(landmarks)

        for n in range(0, 68):
            mark = n
            x = landmarks.part(mark).x
            y = landmarks.part(mark).y

            # Propriedades de transparência
            overlay = image.copy()
            cv2.circle(overlay, (x, y), 10, (255, 255, 255), -1)
            alpha = 0.40
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


            cv2.putText(
                    image,
                    text=str(n),
                    org=(x, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255,255, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )


##### Configuração do MediaPipe

df = pd.DataFrame(columns=["landmark", "x", "y"])

# https://pysource.com/2021/05/14/facial-landmarks-detection-with-opencv-mediapipe-and-python/
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For static images:
IMAGE_FILES = [image_origin]
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
height, width, _ = image_origin.shape


with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    #refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):

        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(
            cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
        )

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
        count = 0

        for face_landmarks in results.multi_face_landmarks:
            for n in lands_mp_match_dlib_682d.landmark_mpipe:

                pt1 = face_landmarks.landmark[n]
                x = int(pt1.x * width)
                y = int(pt1.y * height)

                # procedimento interno Dlib (melhorar código futuramente se necesário, para não precisar prever duas vezes via Dlib)
                landmarks = predictor(gray, rects[0])
                x_dlib = landmarks.part(count).x
                y_dlib = landmarks.part(count).y
                count = count + 1

                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

                cv2.putText(
                    image,
                    text=str(str(n) + ' ('+str(x)+','+str(y)+')'),
                    org=(x, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.3,
                    color=(255, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

                # Desenhando uma linha entre o ponto estimado pelo dlib e o estimado pelo mediapipe
                image = cv2.line(
                    image, (x_dlib, y_dlib), (x, y), (0, 0, 255), thickness=2
                )

                new_row = {
                    "landmark": n,
                    "x": x,
                    "y": y,
                }
                df = df.append(new_row, ignore_index=True)

    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.imwrite("../../data/examples/"+image_name+"_overlaping_682d_teste.jpg", image)
    df.to_csv(
        "../../data/examples/" + image_name + "_overlaping_682d" + str(".csv"),
        sep=";",
    )

