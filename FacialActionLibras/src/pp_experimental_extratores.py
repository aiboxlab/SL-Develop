"""
Nessse script você define os parâmetros do seu pipeline experimental dos extratores
env: FacialActionLibras_mediapipe_vLocal

(FacialActionLibras_mediapipe_vLocal) $ python pp_experimental_extratores.py --save y

"""

from datetime import datetime
import argparse
from config.config import ROOT_DIR
from models.squeezenet_inference_pp_experimental import img_write, neural_net
from utils.utils import create_directory, create_directory_v2, get_all_files_from_directory, tic, tac, read_data_with_subdirectorys
import pandas as pd
import csv

def config_args():
    parser = argparse.ArgumentParser(
        description="Predição de vídeos de uma determinada base"
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
    if write == "y":
        write = True
    else:
        write = False

    return write



def predict_all_db(base, config_experimental):

    write = config_args()

    # Variáveis necessárias

    db_input = "../data/raw/"+base+"/Videos/"
    db_output_frames = "../data/processed/"+base+"-frames/"
    db_output = "../data/outputs/"+base+"/predicts_squeezenet/"
    extension = ".mp4"

    df_results = pd.DataFrame(
            columns=["video_name", "end", "frame", "start", 'aus', 'xy']
        )

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    start_time = tic()
    print(dt_string)

    # Processo de Leitura da Base de Vídeos
    create_directory_v2(db_output)
    create_directory_v2(db_output+'checkpoints/')

    if(write == True):
        create_directory_v2(db_output_frames)##

    lst_videos = read_data_with_subdirectorys(db_input, extension)
    print("Quantidade de Vídeos: " + str(len(lst_videos)))


    for vid in sorted(lst_videos):  # mude para percorrer toda a base [:5]
        split_path = vid.split("/")
        nome_video = split_path[-1].split(".")[0]
        new_path_frames = None
        
        if(write == True):
            new_path_frames = db_output_frames+nome_video+'/'+config_experimental['name_experiment']
            create_directory(new_path_frames) 

        _, df_results = neural_net(
            vid, 
            nome_video, 
            df_results, 
            new_path_frames, 
            write, 
            config_experimental) #path do vídeo, path dos frames dos vídeos

        finished_time_, total_time_, h_, m_, s_ = tac(start_time)
        df_results.to_csv(
            "{0}checkpoint-predicts-{1}-{2}-{3}h{4}m{5}s-{6}.csv".format(db_output+'checkpoints/', base, dt_string, h_, m_, s_, config_experimental['name_experiment']),
            sep=";",
        )

    finished_time, total_time, h, m, s = tac(start_time)

    df_results.to_csv(
        "{0}predicts-{1}-{2}-{3}h{4}m{5}s-{6}.csv".format(db_output, base, dt_string, h, m, s, config_experimental['name_experiment']),
        sep=";",
    )

    print(finished_time, total_time)



def set_configuration(extractor, crop, factor, name_experiment):
    dict = {
        "extractor": extractor,
        "crop": crop,
        "factor": factor,
        "name_experiment": name_experiment
    }
    return dict



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Processo de Extração de Features"
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

    # raw_base = "DISFA-Video_RightCamera-1video"
    raw_base = "HM-Libras"

## DLIB
    """
    crop = True
    factor = 1.15
    extractor = 'dlib'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)

    crop = True
    factor = 1.01
    extractor = 'dlib'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)
    """

    """crop = False
    factor = 1.01
    extractor = 'dlib'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)

    crop = False
    factor = 1.15
    extractor = 'dlib'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)
    """
    """# MEDIAPIPE POSICAO
    crop = True
    factor = 0
    extractor = 'mediapipe_posicao'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)

    crop = True
    factor = 0.025
    extractor = 'mediapipe_posicao'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)

    crop = True
    factor = 0.05
    extractor = 'mediapipe_posicao'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)

    crop = False
    factor = 0
    extractor = 'mediapipe_posicao'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)"""

## MEDIAPIPE ID
    
    """crop = False
    factor = 0
    extractor = 'mediapipe_id'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)"""

    crop = True
    factor = 0.05
    extractor = 'mediapipe_id'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)

    """
    crop = True
    factor = 0
    extractor = 'mediapipe_id'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)

    crop = True
    factor = 0.025
    extractor = 'mediapipe_id'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)

    crop = True
    factor = 0.05
    extractor = 'mediapipe_id'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)
    """

## AUTOENCODER
    """crop = False
    factor = 0
    extractor = 'mediapipe_ae'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)

    crop = True
    factor = 0
    extractor = 'mediapipe_ae'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)

    crop = True
    factor = 0.025
    extractor = 'mediapipe_ae'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)

    crop = True
    factor = 0.05
    extractor = 'mediapipe_ae'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)"""

## MEDIAPIPE ID + CASCADE DETECTOR
    """
    crop = True
    factor = 1.15
    extractor = 'mediapipe_cascade'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)

    crop = True
    factor = 1.01
    extractor = 'mediapipe_cascade'
    config_experimental = set_configuration(extractor, crop, factor, '{}_crop_{}_factor_{}'.format(extractor, crop, str(factor).replace(".", "")))
    predict_all_db(raw_base, config_experimental)
    """