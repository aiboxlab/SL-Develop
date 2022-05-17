"""
----------> ATENÇÃO!!!!!!!!!!!!
ESTE SCRIPT FOI DESCONTINUADO,
PARA REALIZAR PREDIÇÕOS COM O SQUEEZENET USE pp_experimental_extratores.py
E ESCOLHA O EXTRATOR DE FEATURES DESEJADO (mediapipe, dlib) 
O arquivo encontra-se nesse mesmo diretório!"""

from datetime import datetime
import argparse
#from config.config import ROOT_DIR
from FacialActionLibras.src.models.squeezenet_inference import img_write, neural_net
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

def predict_all_db(base, extractor):

    write = config_args()

    # Variáveis necessárias

    db_input = "../data/raw/"+base+"/"
    db_output_frames = "../data/processed/"+base+"-frames/"
    db_output = "../data/outputs/"+base+"/predicts_squeezenet/"
    extension = ".mp4"

    df_results = pd.DataFrame(
            columns=["video_name", "end", "frame", "start", 'aus', 'xy']
        )

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    start_time = tic()

    # Processo de Leitura da Base de Vídeos
    create_directory_v2(db_output)

    if(write == True):
        create_directory_v2(db_output_frames)##

    lst_videos = read_data_with_subdirectorys(db_input, None)
    print("Quantidade de Vídeos: " + str(len(lst_videos)))


    for vid in lst_videos:  # mude para percorrer toda a base [:5]
        split_path = vid.split("/")
        nome_video = split_path[-1].split(".")[0]

        if(write == True):
            create_directory(db_output_frames+nome_video) 

        output, df_results = neural_net(vid, nome_video, df_results, db_output_frames+nome_video, write, extractor) #path do vídeo, path dos frames dos vídeos

        """output.write(
            "{0}{1}-{2}.xml".format(db_output, nome_video, dt_string),
            encoding="utf-8",
            xml_declaration=True,
        )"""

        """df_results.to_csv(
            "{0}{1}-{2}.csv".format(db_output, nome_video, dt_string),
            sep=";",
        )"""

    df_results.to_csv(
        "{0}predicts-{1}-{2}_{3}.csv".format(db_output, base, dt_string, extractor),
        sep=";",
    )

    finished_time, total_time = tac(start_time)
    print(finished_time, total_time)



def make_model_inference():
    files_name = get_all_files_from_directory("../data/raw/vamos_executar/*.avi")
    start_time = tic()
    
    for file in files_name:
        print("#####################################")
        print(file)
        _f = file.split(".avi")
        _f = _f[0].split("/")
        image_path = f"../data/examples/images/{_f[-1]}"
        create_directory(image_path)
        dt = datetime.now()
        dt_str = dt.strftime("%d-%m-%Y-%H-%M-%S")
  
        print(file, image_path)
        print(f"{ROOT_DIR}/outputs/{_f[-1]}-{dt_str}.xml")
        output = neural_net(file, image_path)
        
        output.write(
            f"{ROOT_DIR}/outputs/{_f[-1]}-{dt_str}.xml",
            encoding="utf-8",
            xml_declaration=True,
        )
        finished_time, total_time = tac(start_time)
        print(finished_time, total_time)



if __name__ == "__main__":
    #get_face_frames()
    #make_model_inference()
        
    raw_base = "teste-mari"#"Videos-Emely-Extra"
    extractor = 'mediapipe'
    predict_all_db(raw_base, extractor)

