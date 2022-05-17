# https://medium.com/brasil-ai/mapeamento-facial-landmarks-com-dlib-python-3a200bb35b87

# import the necessary packages
import cv2
import dlib
import pandas as pd
from imutils import face_utils

# https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
# FACIAL_LANDMARKS_IDXS = OrderedDict([
# ("mouth", (48, 68)),
# ("right_eyebrow", (17, 22)),
# ("left_eyebrow", (22, 27)),
# ("right_eye", (36, 42)),
# ("left_eye", (42, 48)),
# ("nose", (27, 35)),
# ("jaw", (0, 17))
# ])


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()
    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [
            (19, 199, 109),
            (79, 76, 240),
            (230, 159, 23),
            (168, 100, 168),
            (158, 163, 32),
            (163, 38, 32),
            (180, 42, 220),
        ]


# Vamos inicializar um detector de faces (HOG) para então
# fazer a predição dos pontos da nossa face.
# p é o diretorio do nosso modelo já treinado, no caso, ele está no mesmo diretorio
# que esse script
p = "../../models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

df_keys = pd.DataFrame(columns=["frame", "video_name", "keys"])

video_output = "dlib-jayne-webcam.avi"
video_input = "jayne-webcam.avi"
video_name = video_input.split("/")[-1]

cap = cv2.VideoCapture(video_input)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter(
    "dlib-jayne-webcam.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, size
)

if cap.isOpened() == False:
    print("Error opening video stream or file")

while cap.isOpened():
    # Obtendo vídeo e transformando-o preto e branco.
    _, image = cap.read()

    if _ == True:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectando as faces em preto e branco.
        rects = detector(gray, 0)

        # para cada face encontrada, encontre os pontos de interesse.
        for (i, rect) in enumerate(rects):
            # faça a predição e então transforme isso em um array do numpy.
            shape = predictor(gray, rect)

            shape = face_utils.shape_to_np(shape)

            # desenhe na imagem cada cordenada(x,y) que foi encontrado.
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            new_row = {
                "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                "video_name": video_name,
                "keys": shape,
            }

            print("----------------------")
            print(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(shape)

            df_keys = df_keys.append(new_row, ignore_index=True)

        result.write(image)
        # Mostre a imagem com os pontos de interesse.
        cv2.imshow("../../../data/processed/video", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    else:
        break

df_keys.to_csv(
    str(video_output) + str(video_name.split(".")[0]) + str(".csv"), sep=";"
)
cap.release()
result.release()
cv2.destroyAllWindows()
