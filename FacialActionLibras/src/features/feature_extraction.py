import glob
import json
import os
import sys

import dlib
import cv2
import dlib
import numpy as np
import pandas as pd
import utils
from imutils import face_utils
from protobuf_to_dict import protobuf_to_dict  # pip3 install protobuf3_to_dict
import numpy as np
import subprocess



class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def dlib_feature_extraction(
    video_input,
    video_output,
    detector,
    predictor,
    df_keys,
    video_name,
    write=None,
):

    cap = cv2.VideoCapture(video_input)

    if write:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        result = cv2.VideoWriter(
            video_output + video_name,
            cv2.VideoWriter_fourcc(*"MJPG"),
            25,
            size,
        )

    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while cap.isOpened():
        # Obtendo vídeo e transformando-o preto e branco.
        success, image = cap.read()

        if not success:
            break

        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = image

        # Detectando as faces em preto e branco.
        rects = detector(gray, 0)

        shape = 0

        # para cada face encontrada, encontre os pontos de interesse.
        for (i, rect) in enumerate(rects):
            # faça a predição e então transforme isso em um array do numpy.
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # desenhe na imagem cada cordenada(x,y) que foi encontrado.
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            lists = shape.tolist()
            keys = json.dumps(lists)

            new_row = {
                "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                "video_name": video_name,
                "keys": keys,
            }

            # print("----------------------")
            # print(video_name)
            # print(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # print(json_str)

            # df_keys = df_keys.append(new_row, ignore_index=True)

            print(bcolors.OKGREEN + str("DETECTED " + str(shape.shape)))

        if len(rects) == 0:
            print(bcolors.FAIL + str("*** NOT DETECTED ***"))
            # print("NOOOOO5555")
            # arr_external = np.array([[0,0]])
            new_row = {
                "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                "video_name": video_name,
                "keys": None,
            }

        print("------------------------")
        print(
            "Frame "
            + str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            + str(" " + str(video_input))
        )

        df_keys = df_keys.append(new_row, ignore_index=True)
        # cv2.imshow("../../../data/processed/examples", image)
        if write:
            result.write(image)
            cv2.imshow("Dlib", image)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    return df_keys
    # df_keys.to_csv(str(video_output) + str(video_name.split(".")[0])+ str(".csv"), sep=';')
    cv2.destroyAllWindows()
    cap.release()


def mediapipe_feature_extraction(
    video_input,
    video_output,
    mp_drawing,
    mp_drawing_styles,
    mp_face_mesh,
    df_keys,
    video_name,
    write=None,
):

    cap = cv2.VideoCapture(video_input)

    if write:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        result = cv2.VideoWriter(
            video_output + video_name,
            cv2.VideoWriter_fourcc(*"MJPG"),
            25,
            size,
        )

    if cap.isOpened() == False:
        print("Error opening video stream or file")

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        #refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        while cap.isOpened():
            success, image = cap.read()
            if success:
                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                        )
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                        )

                        keypoints = protobuf_to_dict(face_landmarks)
                        # arr_external = np.array([[0,0,0]])

                        arr_external = np.array(
                            [
                                [
                                    keypoints["landmark"][0]["x"],
                                    keypoints["landmark"][0]["y"],
                                    keypoints["landmark"][0]["z"],
                                ]
                            ]
                        )

                        for i in range(1, len(keypoints["landmark"])):
                            arr_internal_ = np.array(
                                [
                                    [
                                        keypoints["landmark"][i]["x"],
                                        keypoints["landmark"][i]["y"],
                                        keypoints["landmark"][i]["z"],
                                    ]
                                ]
                            )

                            arr_external = np.concatenate(
                                (arr_external, arr_internal_), axis=0
                            )

                        lists = arr_external.tolist()
                        keys = json.dumps(lists)

                        new_row = {
                            "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                            "video_name": video_name,
                            "keys": keys,
                        }
                    print(
                        bcolors.OKGREEN
                        + str("DETECTED " + str(arr_external.shape))
                    )
                else:
                    print(bcolors.FAIL + str("*** NOT DETECTED ***"))
                    arr_external = np.array([[0, 0, 0]])
                    new_row = {
                        "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                        "video_name": video_name,
                        "keys": None,
                    }

                print("------------------------")
                print(
                    "Frame "
                    + str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                    + str(" " + str(video_input))
                )
                # print("Qt Keys "+str(len(keypoints['landmark'])))

                df_keys = df_keys.append(new_row, ignore_index=True)

                # print(arr_external)
                # print(arr_external.shape)
                if write:
                    result.write(image)
                    cv2.imshow("MediaPipe FaceMesh", image)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
            else:
                break

        return df_keys
        cv2.destroyAllWindows()
        cap.release()


def openface_feature_extraction(video_input, video_output):

    retorno = subprocess.run(
        [
            "./../../models/openface/build/bin/FeatureExtraction2Way",
            "-f",
            video_input,
            "-out_dir",
            video_output,
            "-2Dfp",
        ],
    )

    print("\n ------> Vídeo {}\n".format(video_input))
    print("\n ------> Output{}\n".format(video_output))
    return 1
