"""
Para execução: python scaling_crop_resized_frames.py
Env: FacialActionLibras_vLocal
"""

import cv2, numpy as np
import os
import pandas as pd
import math
from datetime import datetime
import utils as u


def crop_slices(image, height, width, slice_size):
    return image[0:height, 0+slice_size:width-slice_size]


def resize_square(image, resolution_factor):
    return cv2.resize(image, (resolution_factor, resolution_factor)) 


def load_video(path_input, path_save, to_resolution):

    cap = cv2.VideoCapture(path_input)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (to_resolution, to_resolution)
    result = 0

    result = cv2.VideoWriter(
        str(path_save),
        #cv2.VideoWriter_fourcc(*"MJPG"), #avi
        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), #mp4
        #cv2.VideoWriter_fourcc(*'MP4V'),
        fps,
        size,
    )

    if cap.isOpened() == False:
        print("Error opening video stream or file")

    return cap, fps, result


def make_frame(image, to_resolution):
    height, width, _ = image.shape

    #Difereça entre as dimensões de largura e altura (tamanho dopedaço que será cortado)
    diff = max(height, width) - min(height, width)

    #Vamos dividir o "pedaço" anterior em dois pedaços de igual tamanho (para remover 1 da esquerda e outro da direita)
    slice_size= int(math.floor(diff/2))

    #Vamos cropar o "pedaço" esquedo e o "pedaço" direito
    crop_img = crop_slices(image, height, width, slice_size)#image[0:height, 0+slice_size:width-slice_size]
    #image = cv2.rectangle(image, (0+slice_size, 0), (width-slice_size, height), (255, 0, 0), 2) #Proporção blue

    #Após termos uma imagem quadrada, iremos redimensionar
    resize_img = resize_square(crop_img, to_resolution)

    #resize_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)

    #print(resize_img.shape)

    return resize_img


def make_video(path_input, path_save, to_resolution):
    
    cap, fps, result = load_video(path_input, path_save, to_resolution)

    while cap.isOpened():
        _, image = cap.read()

        if _ == True:
            frame_resized = make_frame(image, to_resolution)
            result.write(frame_resized)
            
        else:
            break

    cv2.destroyAllWindows()
    cap.release()
    result.release()



def make_database(base, db_input, db_output, extension, to_resolution):

    # Variáveis necessárias
    #base = "HM-Libras"
    #db_input = "../../data/raw/"+base+"/Videos/"
    #db_output = "../../data/processed/"+base+"-augmentation/"

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    start_time = u.tic()

    # Processo de Leitura da Base de Vídeos
    u.create_directory(db_output)
    lst_videos = u.read_data_with_subdirectorys(db_input, extension)
    print("Quantidade de Vídeos: " + str(len(lst_videos)))
    count = 1

    for vid in lst_videos:  # mude para percorrer toda a base [:5]
        split_path = vid.split("/")
        nome_video = split_path[-1]
        classe_video = split_path[-2]
        path_save = None

        print('::: Vídeo: {0}, Classe: {1} :::'.format(nome_video, classe_video))
        
        path_save = u.create_directory(db_output+"/")

        raw_video = None
        path_save = path_save + nome_video

        vid = make_video(vid, path_save, to_resolution)

    finished_time, total_time = u.tac(start_time)

    print(finished_time, total_time)


if __name__ == "__main__":
    
    base = "HM-Libras"
    db_input = "../../data/raw/"+base+"/Videos"
    db_output = "../../data/processed/"+base+"_crop-resized/"
    extension = '.mp4'
    to_resolution = 224

    make_database(base, db_input, db_output, extension, to_resolution)
