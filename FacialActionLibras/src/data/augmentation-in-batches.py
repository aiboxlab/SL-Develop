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


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.compat.v1.Session(config=config)

print(tf.__version__)
#FPS = 30

def get_fps(cap):
    #v_entry = cv2.VideoCapture(video_path, 0)
    #frames = int(v_entry.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps

def preprocessing(vid):
    metadata = media.read_video(vid)
    print(type(metadata))
    print(metadata.shape)
    raw_video = media.to_float01(metadata)
    print(type(raw_video))
    print(raw_video.shape)

    return raw_video, metadata.metadata.fps



def preprocessing_v2(path_input, path_save, func):

    cap = cv2.VideoCapture(path_input)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = 0

    if(func == 'rgb_to_grayscale' ):
        result = cv2.VideoWriter(
        #path_save, cv2.VideoWriter_fourcc('*"MJPG"'), 30, size, isColor=False #for codec origin MPEG-4 Video
        path_save, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size, isColor=False #for codec origin JPEG
        )

    else:
        result = cv2.VideoWriter(
            str(path_save),
            #cv2.VideoWriter_fourcc(*"MJPG"), #avi
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), #mp4
            fps,
            size,
        )

    if cap.isOpened() == False:
        print("Error opening video stream or file")

    return cap, fps, result


################ Início do bloco de funções que geram o augmentation dos vídeos ################

# class Augmentation():
def resize(path_input, path_save, func):
    """with tf.device("/gpu:0"):
        resized = tf.image.resize(video, size=(240, 240))
        media.write_video(path_save, video.numpy(), fps=get_fps(video))
    return resized"""

    cap, fps, result = preprocessing_v2(path_input, path_save, func)

    while cap.isOpened():
        _, image = cap.read()

        if _ == True:
            image =  tf.image.resize(image, size=(480, 480)).numpy()
            result.write(image)
        else:
            break


def rot90(path_input, path_save, func):

    cap, fps, result = preprocessing_v2(path_input, path_save, func)

    while cap.isOpened():
        _, image = cap.read()

        if _ == True:
            image =  tf.image.resize(image, size=(480, 480)).numpy()
            image =  tf.image.rot90(image, k=1).numpy()
            result.write(image)
        else:
            break


def adjust_brightness(path_input, path_save, func):

    cap, fps, result = preprocessing_v2(path_input, path_save, func)

    while cap.isOpened():
        _, image = cap.read()

        if _ == True:
            image =  tf.image.adjust_brightness(image, delta=0.5).numpy()
            result.write(image)
        else:
            break


def adjust_contrast(path_input, path_save, func):

    cap, fps, result = preprocessing_v2(path_input, path_save, func)

    while cap.isOpened():
        _, image = cap.read()

        if _ == True:
            image =  tf.image.adjust_contrast(image, contrast_factor=0.25).numpy()
            result.write(image)
        else:
            break


def adjust_hue(path_input, path_save, func):


    cap, fps, result = preprocessing_v2(path_input, path_save, func)

    while cap.isOpened():
        _, image = cap.read()

        if _ == True:
            image =  tf.image.adjust_hue(image, delta=1.5).numpy()
            result.write(image)
        else:
            break


def stateless_random_brightness_95(path_input, path_save, func):


    cap, fps, result = preprocessing_v2(path_input, path_save, func)

    while cap.isOpened():
        _, image = cap.read()

        if _ == True:
            image =  tf.image.stateless_random_brightness(image, max_delta=0.95, seed=(0, 0)).numpy()
            result.write(image)
        else:
            break

def stateless_random_brightness_50(path_input, path_save, func):


    cap, fps, result = preprocessing_v2(path_input, path_save, func)

    while cap.isOpened():
        _, image = cap.read()

        if _ == True:
            image =  tf.image.stateless_random_brightness(image, max_delta=0.5, seed=(0, 0)).numpy()
            result.write(image)
        else:
            break

def stateless_random_brightness_30(path_input, path_save, func):


    cap, fps, result = preprocessing_v2(path_input, path_save, func)

    while cap.isOpened():
        _, image = cap.read()

        if _ == True:
            image =  tf.image.stateless_random_brightness(image, max_delta=0.3, seed=(0, 0)).numpy()
            result.write(image)
        else:
            break

def rgb_to_grayscale(path_input, path_save, func):
    
    cap, fps, result = preprocessing_v2(path_input, path_save, func)


    while cap.isOpened():
        _, image = cap.read()

        if _ == True:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result.write(gray)
            
        else:
            break

    cv2.destroyAllWindows()
    cap.release()
    result.release()



def adjust_jpeg_quality_10(path_input, path_save, func):

    cap, fps, result = preprocessing_v2(path_input, path_save, func)

    while cap.isOpened():
        _, image = cap.read()

        if _ == True:
            image = tf.image.adjust_jpeg_quality(image, 10).numpy()
            result.write(image)
        else:
            break

def adjust_jpeg_quality_15(path_input, path_save, func):

    cap, fps, result = preprocessing_v2(path_input, path_save, func)

    while cap.isOpened():
        _, image = cap.read()

        if _ == True:
            image = tf.image.adjust_jpeg_quality(image, 15).numpy()
            result.write(image)
        else:
            break

################ Início do bloco de funções que geram o augmentation dos vídeos ################


def make_augumentation():

    # Variáveis necessárias
    base = "HM-Libras"
    db_input = "../../data/raw/"+base+"/Videos/"
    db_output = "../../data/processed/"+base+"-augmentation/"

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    start_time = u.tic()

    # Processo de Leitura da Base de Vídeos
    u.create_directory(db_output)
    lst_videos = u.read_data_with_subdirectorys(db_input, '.mp4')
    print("Quantidade de Vídeos: " + str(len(lst_videos)))
    count = 1

    functions = {
        ##"resize: resize,
        ##"rot90": rot90,
        "adjust_brightness": adjust_brightness,
        "adjust_contrast": adjust_contrast,
        "adjust_hue": adjust_hue,
        "stateless_random_brightness_95": stateless_random_brightness_95,
        "adjust_jpeg_quality_10": adjust_jpeg_quality_10,
        "adjust_jpeg_quality_15": adjust_jpeg_quality_15,
        "rgb_to_grayscale": rgb_to_grayscale,#1hour:35min:16sec
        "stateless_random_brightness_50":stateless_random_brightness_50, #0hour:13min:39sec
        #"stateless_random_brightness_30":stateless_random_brightness_30
    }

    for func in functions:
        for vid in lst_videos:  # mude para percorrer toda a base [:5]
            split_path = vid.split("/")
            nome_video = split_path[-1]
            classe_video = split_path[-2]
            path_save = None

            print('::: Função: {2}, Vídeo: {0}, Classe: {1} :::'.format(nome_video, classe_video, func))
            


            #raw_video = preprocessing(vid)

            #cap, fps = preprocessing_v2(vid)

            



            if(classe_video == base):
                path_save = u.create_directory(
                    db_output + func + "/"
                )
            else:

                path_save = u.create_directory(
                    db_output + func + "/" + classe_video + "/"
                )


            raw_video = None
            path_save = path_save + nome_video

            vid = functions[func](vid, path_save, func)

            tf.keras.backend.clear_session()

    finished_time, total_time = u.tac(start_time)
    # logging.info('Finished')
    # logging.info(total_time)

    print(finished_time, total_time)


if __name__ == "__main__":
    make_augumentation()


# 0hour:47min:53sec
