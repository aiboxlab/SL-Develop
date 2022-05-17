"""
Configuração de Env:
	Mude o MODE (DLIB ou MEDIAPIPE)
	Mude o env anaconda (mediapipe39 ou dlib)

Descrição:
	Este script é responsável por realizar a extração de features, 
	conforme o modo de extrator escolhido

Exemplo de execução:
	python main.py --mode mediapipe
"""

import dlib
import cv2
import pandas as pd
import sys
import os
import glob
from imutils import face_utils
import json
import feature_extraction as fe
import utils as u
import argparse
import logging
from datetime import datetime
import time

# user_path = os.path.expanduser('~')
# path = str(user_path) + "../../data/raw/SinaisLibras-Tamires/Videos/"
# path = "../../data/raw/SinaisLibras-Tamires/" Use esse quando o DVC estiver configurarado para múltipĺos diretórios


def make_extraction():
    # try:
    # Configuração de input
    parser = argparse.ArgumentParser(
        description="Processo de Extração de Features"
    )
    parser.add_argument(
        "--mode",
        action="store",
        dest="mode",
        required=True,
        help="Modo de extração de features.",
    )
    parser.add_argument(
        "--save",
        action="store",
        dest="save",
        required=False,
        help="Armazenamento de vídeo com landmarks.",
    )
    arguments = parser.parse_args()

    # Arg SAVE
    save = arguments.save
    if (save != "y") and (save != "n") and (save != None):
        print("[!] Para operar use...\n [y] para gravar\n [n] para não gravar")
        exit()

    write = arguments.save
    print("Save: " + str(write))
    if write == "y":
        write = True
    else:
        write = False

    # Arg MODE
    modo = arguments.mode
    print("MODO: " + str(modo))
    if (modo != "dlib") and (modo != "mediapipe"):
        print("[!] Para operar use...\n [dlib]\n [mediapipe]")
        exit()

    path_bases = "../../data/processed/UCF-101-Analysis-augmentation/"
    list_augs = os.listdir(path_bases)

    for i in list_augs:

        # Variáveis necessárias
        base = i + "/"
        path = path_bases + str(base)
        video_output_dlib = (
            "../../data/processed/dlib/UCF-101-Analysis-augmentation/"
            + str(base)
        )
        video_output_mediapipe = (
            "../../data/processed/mediapipe/UCF-101-Analysis-augmentation/"
            + str(base)
        )
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H%M%S")

        # Configuração de modo (dlib ou mediapipe)
        if modo == "dlib":
            log_dir = u.create_directory(video_output_dlib + "log/")
            logging.basicConfig(
                filename=(
                    log_dir + "feature_extraction_" + str(dt_string) + ".log"
                ),
                filemode="w",
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                level=logging.INFO,
            )

            logging.info("Started")
            logging.info("From... " + path)
            logging.info("To... " + video_output_dlib)

            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(
                "../../models/shape_predictor_68_face_landmarks.dat"
            )
            video_output = video_output_dlib

        elif modo == "mediapipe":
            log_dir = u.create_directory(video_output_mediapipe + "log/")
            logging.basicConfig(
                filename=(
                    log_dir + "feature_extraction_" + str(dt_string) + ".log"
                ),
                filemode="w",
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                level=logging.INFO,
            )
            logging.info("Started")
            logging.info("From... " + path)
            logging.info("To... " + video_output_mediapipe)

            import mediapipe as mp

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_face_mesh = mp.solutions.face_mesh
            video_output = video_output_mediapipe

        start_time = u.tic()

        # Processo de Leitura da Base de Vídeos
        u.create_directory(video_output)
        lst_videos = u.read_data_with_subdirectorys(path)
        print("Quantidade de Vídeos: " + str(len(lst_videos)))
        count = 1

        # Processo de Extração das Features da Base de Vídeos
        for i in lst_videos:  # mude para percorrer toda a base [:5]

            df_keys = pd.DataFrame(
                columns=["frame", "video_name", "keys"]
            )  # Arquivo que contém as features
            video_name = i.split("/")[-1]
            print(str(count) + "/" + str(len(lst_videos)) + " " + i)

            if modo == "dlib":
                df_keys = fe.dlib_feature_extraction(
                    i,
                    video_output,
                    detector,
                    predictor,
                    df_keys,
                    video_name,
                    write,
                )
            elif modo == "mediapipe":
                df_keys = fe.mediapipe_feature_extraction(
                    i,
                    video_output,
                    mp_drawing,
                    mp_drawing_styles,
                    mp_face_mesh,
                    df_keys,
                    video_name,
                    write,
                )

            df_keys.to_csv(
                str(video_output)
                + str(video_name.split(".")[0])
                + str(".csv"),
                sep=";",
            )
            count = count + 1

        finished_time, total_time = u.tac(start_time)
        logging.info("Finished")
        logging.info(total_time)

        # except Exception as e:
        # logging.error("Exception occurred", exc_info=True)


if __name__ == "__main__":
    make_extraction()
