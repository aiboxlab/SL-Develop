"""
Geração de data augmentation de uma determinada base de vídeos
"""

import cv2
import pandas as pd
import sys
import os
import glob
import json
import utils as u
import argparse
import logging
from datetime import datetime
import time
import mediapy as media
import numpy as np
import tensorflow as tf



print(tf.__version__)
FPS = 30

def get_fps(video_path):
    v_entry = cv2.VideoCapture(video_path, 0)
    #frames = int(v_entry.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = v_entry.get(cv2.CAP_PROP_FPS)
    return fps

def preprocessing(vid):
    raw_video = media.read_video(vid)
    raw_video = media.to_float01(raw_video)
    return raw_video


################ Início do bloco de funções que geram o augmentation dos vídeos ################

# class Augmentation():
def resize(video, path_save, path_input=None):
    with tf.device("/gpu:0"):
        resized = tf.image.resize(video, size=(240, 240))
        media.write_video(path_save, video.numpy(), fps=FPS)
    return resized


def rot90(video, path_save, path_input=None):
    with tf.device("/gpu:0"):
        rot90 = tf.image.rot90(video, k=1)
        print(rot90.shape)
        #cap = cv2.VideoCapture(rot90)[0]
        media.write_video(path_save, rot90.numpy(), fps=FPS)
    return rot90


def adjust_brightness(video, path_save, path_input=None):
    with tf.device("/gpu:0"):
        adjust_brightness = tf.image.adjust_brightness(
            video, delta=0.5
        )  # [0,1)
        media.write_video(path_save, adjust_brightness.numpy(), fps=FPS)
    return adjust_brightness


def adjust_contrast(video, path_save, path_input=None):
    with tf.device("/gpu:0"):
        adjust_contrast = tf.image.adjust_contrast(
            video, contrast_factor=0.25
        )  # um multiplicador float
        media.write_video(path_save, adjust_contrast.numpy(), fps=FPS)
    return adjust_contrast


def adjust_hue(video, path_save, path_input=None):
    with tf.device("/gpu:0"):
        adjust_hue = tf.image.adjust_hue(
            video, delta=1.5
        )  # [-1, 1] #Quanto adicionar ao canal de matiz.
        media.write_video(path_save, adjust_hue.numpy(), fps=FPS)
    return adjust_hue


def stateless_random_brightness(video, path_save, path_input=None):
    with tf.device("/gpu:0"):
        stateless_random_brightness = tf.image.stateless_random_brightness(
            video, max_delta=0.95, seed=(0, 0)
        )  # float, must be non-negative. Garante os mesmos resultados dados os mesmos seedindependentemente de quantas vezes a função é chamada
        media.write_video(
            path_save, stateless_random_brightness.numpy(), fps=FPS
        )
    return stateless_random_brightness


def adjust_jpeg_quality(video, path_save, path_input=None):
    with tf.device("/gpu:0"):
        adjust_jpeg_quality = tf.map_fn(
            fn=lambda t: tf.image.adjust_jpeg_quality(t, 10),
            elems=tf.constant(video),
        )
        media.write_video(path_save, adjust_jpeg_quality.numpy(), fps=FPS)
    return adjust_jpeg_quality


"""def adjust_fps_rate_30(video, path_save):
	with tf.device('/gpu:0'): 
		media.write_video(path_save, video, fps=30)
	return adjust_fps_rate_30"""


def rgb_to_grayscale(video, path_save, path_input):
    cap = cv2.VideoCapture(path_input)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = 0
    result = cv2.VideoWriter(
        #path_save, cv2.VideoWriter_fourcc('*"MJPG"'), 30, size, isColor=False #for codec origin MPEG-4 Video
        path_save, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, size, isColor=False #for codec origin JPEG
    )

    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while cap.isOpened():
        _, image = cap.read()

        if _ == True:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result.write(gray)
        else:
            break

    cv2.destroyAllWindows()
    cap.release()

    # return result


################ Início do bloco de funções que geram o augmentation dos vídeos ################


def video_augmentation(video_numpy):
    with tf.device("/gpu:0"):
        list_augs = [
            #"resized",
            "bright",
            "contrast",
            "hue_adjust",
            #"rot90",
            "stateless_random_brightness",
            "adjust_jpeg_quality",
            "adjust_fps_rate",
        ]

        resized = video_numpy#tf.image.resize(video_numpy, size=(240, 240))

        bright = tf.image.adjust_brightness(resized, delta=0.5)  # [0,1)
        contrast = tf.image.adjust_contrast(
            resized, contrast_factor=0.25
        )  # um multiplicador float
        hue_adjust = tf.image.adjust_hue(
            resized, delta=1.5
        )  # [-1, 1] #Quanto adicionar ao canal de matiz.
        #rot90 = tf.image.rot90(
        #    resized, k=1
        #)  # O número de vezes que a imagem é girada em 90 graus
        stateless_random_brightness = tf.image.stateless_random_brightness(
            resized, max_delta=0.95, seed=(0, 0)
        )  # float, must be non-negative. Garante os mesmos resultados dados os mesmos seedindependentemente de quantas vezes a função é chamada
        adjust_jpeg_quality = tf.map_fn(
            fn=lambda t: tf.image.adjust_jpeg_quality(t, 10),
            elems=tf.constant(resized),
        )  # [0, 100]
        # Fps max

        stacked = tf.stack(
            [
                #resized,
                bright,
                contrast,
                hue_adjust,
                #rot90,
                stateless_random_brightness,
                adjust_jpeg_quality.numpy(),
            ]
        )

        return list_augs, stacked


def make_augumentation():

    # Variáveis necessárias
    base = "DISFA"
    db_input = "../../data/raw/"+base+"/"
    db_output = "../../data/processed/"+base+"-augmentation/"

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    start_time = u.tic()

    # Processo de Leitura da Base de Vídeos
    u.create_directory(db_output)
    lst_videos = u.read_data_with_subdirectorys(db_input, '.avi')
    print("Quantidade de Vídeos: " + str(len(lst_videos)))
    count = 1

    functions = {
        #'resize': resize,
        #"rot90": rot90,
        "adjust_brightness": adjust_brightness,
        "adjust_contrast": adjust_contrast,
        "adjust_hue": adjust_hue,
        "stateless_random_brightness": stateless_random_brightness,
        "adjust_jpeg_quality": adjust_jpeg_quality,
        "rgb_to_grayscale": rgb_to_grayscale,
    }

    for func in functions:
        print(func)

        for vid in lst_videos:  # mude para percorrer toda a base [:5]
            split_path = vid.split("/")
            nome_video = split_path[-1]
            classe_video = split_path[-2]
            print(nome_video)
            print(classe_video)

            raw_video = preprocessing(vid)

            path_save = u.create_directory(
                db_output + func + "/" + classe_video + "/"
            )

            print(raw_video.shape)
            path_save = path_save + nome_video
            #vid = functions[func](raw_video[:64], path_save, vid)
            vid = functions[func](raw_video, path_save, vid)

            tf.keras.backend.clear_session()

    finished_time, total_time = u.tac(start_time)
    # logging.info('Finished')
    # logging.info(total_time)

    print(finished_time, total_time)


if __name__ == "__main__":
    make_augumentation()


# 0hour:47min:53sec
