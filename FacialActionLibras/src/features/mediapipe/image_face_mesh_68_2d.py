import cv2, numpy as np
import mediapipe as mp
import os
import pandas as pd
from protobuf_to_dict import protobuf_to_dict

# https://pysource.com/2021/05/14/facial-landmarks-detection-with-opencv-mediapipe-and-python/

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

image_name = "machado_assis"
lands_mp_match_dlib_682d = pd.read_csv(
    "../../../data/examples/lands_mp_match_dlib_682d_ref02.csv", sep=";"
)
image = cv2.imread("../../../data/examples/" + image_name + "_lands_dlib.jpg")

# For static images:
IMAGE_FILES = [image]
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
height, width, _ = image.shape

df = pd.DataFrame(columns=["landmark", "x", "y"])

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
        max_num_faces=1,
        #refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    print("222")
    for idx, file in enumerate(IMAGE_FILES):
        print("333")
        # image = cv2.imread(file)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
        # annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            # print("face_landmarks:", face_landmarks)
            """mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )"""

            """keypoints = protobuf_to_dict(face_landmarks)
            for i in range(0, len(keypoints["landmark"])):
                print(i, keypoints["landmark"][i]["x"],
                      keypoints["landmark"][i]["y"])"""

            for n in lands_mp_match_dlib_682d.landmark_mpipe:

                pt1 = face_landmarks.landmark[n - 1]
                x = int(pt1.x * width)
                y = int(pt1.y * height)

                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
                """mark = n
                x = keypoints["landmark"][n]["x"]
                y = keypoints["landmark"][n]["y"]
                cv2.circle(image, (x, y), 4, (0, 0, 255), -1)"""

                cv2.putText(
                    image,
                    text=str(n + 1),
                    org=(x, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.3,
                    color=(255, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

                new_row = {
                    "landmark": n + 1,
                    "x": x,
                    "y": y,
                }
                df = df.append(new_row, ignore_index=True)

        # print("Qt Keys " + str(len(keypoints["landmark"])))

    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.imwrite(
        "../../../data/examples/" + image_name + "_landm_mediap_682d.jpg",
        image,
    )
    df.to_csv(
        "../../../data/examples/"
        + image_name
        + "_landm_mediap_682d"
        + str(".csv"),
        sep=";",
    )
