# https://medium.com/brasil-ai/mapeamento-facial-landmarks-com-dlib-python-3a200bb35b87

# import the necessary packages
import json
import dlib
import cv2
import numpy as np
import pandas as pd
import json
from imutils import face_utils


# Vamos inicializar um detector de faces (HOG) para então
# fazer a predição dos pontos da nossa face.
# p é o diretorio do nosso modelo já treinado, no caso, ele está no mesmo diretorio
# que esse script
p = "../../../models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

video_output = "../../../data/processed/examples/"
df_keys = pd.DataFrame(columns=["frame", "video_name", "keys"])

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

result = cv2.VideoWriter(
    str(video_output) + "dlib-jayne-webcam.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    30,
    size,
)

count = 0

if cap.isOpened() == False:
    print("Error opening video stream or file")

while cap.isOpened():
    # Obtendo nossa imagem através da webCam e transformando-a preto e branco.
    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectando as faces em preto e branco.
    rects = detector(gray, 0)

    shape = 0
    # arr_external = np.array([[0,0]])

    # para cada face encontrada, encontre os pontos de interesse.
    for (i, rect) in enumerate(rects):
        # faça a predição e então transforme isso em um array do numpy.
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Caso seja a primeira iteração, inicialize o array com o primeiro valor
        # arr_internal_ = shape
        # if(len(arr_external) > 1):
        # arr_external = np.concatenate((arr_external, arr_internal_), axis=0)
        # else:
        # arr_external = arr_internal_

        # print(arr_external[[0]])
        # print(arr_external[[1]])
        # print(arr_external[[2]])
        # print(arr_external[[3]])
        # print(arr_external[[4]])

        print(shape.shape)
        print("-------------------------")

        # desenhe na imagem cada cordenada(x,y) que foi encontrado.
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        lists = shape.tolist()
        keys = json.dumps(lists)

        new_row = {
            "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            "video_name": "teste_webcam_write_dlib",
            "keys": keys,
        }

    if len(rects) == 0:
        print("NOOOOO5555")
        # arr_external = np.array([[0,0]])
        new_row = {
            "frame": int(result.get(cv2.CAP_PROP_POS_FRAMES)),
            "video_name": "teste_webcam_write",
            "keys": None,
        }

    print("Frame " + str(count))

    result.write(image)
    # Mostre a imagem com os pontos de interesse.
    cv2.imshow("../../../data/processed/examples", image)
    count = count + 1

    df_keys = df_keys.append(new_row, ignore_index=True)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

df_keys.to_csv(
    str(video_output) + str("teste_webcam_write-dlib") + str(".csv"), sep=";"
)

cap.release()
result.release()
cv2.destroyAllWindows()
