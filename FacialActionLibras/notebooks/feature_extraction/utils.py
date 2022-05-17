import os
import glob
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
import cv2


def pre_processed(inputs, label_augmentation):

    """
    **DESCRIÇÃO:
    Essa função é responsável por coletar todos os arquivos de features
    de cada vídeo da base submetida aos extratores e concatená-los em um único arquivo.

    **INPUT:
    inputs: dicionário de extrator vs path dos dados preprocessados csv.
    label_augmentation: label do nome da técnica de augmentation utilizada.

    **OUTPUT:
    combined_csv: Arquivo csv com as informações dos extratores concatenados.

    """

    print(inputs["dlib"])
    extension = "csv"

    label_dlib = "dlib" + label_augmentation
    label_mediapipe = "mediapipe" + label_augmentation
    print("----- Pré concat -----")

    # Dlib
    all_filenames = [
        i for i in glob.glob(inputs["dlib"] + "*.{}".format(extension))
    ]
    combined_csv_dlib = pd.concat(
        [pd.read_csv(f, sep=";", index_col=None) for f in all_filenames]
    )
    combined_csv_dlib["extractor"] = label_dlib
    print("Quantidade de Vídeos [Dlib]: " + str(len(all_filenames)))
    print("Quantidade de Frames [Dlib]: " + str(combined_csv_dlib.shape[0]))

    # Mediapipe
    all_filenames = [
        i for i in glob.glob(inputs["mediapipe"] + "*.{}".format(extension))
    ]
    combined_csv_mediapipe = pd.concat(
        [pd.read_csv(f, sep=";", index_col=None) for f in all_filenames]
    )
    combined_csv_mediapipe["extractor"] = label_mediapipe
    print("Quantidade de Vídeos [Mediapipe]: " + str(len(all_filenames)))
    print(
        "Quantidade de Frames [Mediapipe]: "
        + str(combined_csv_mediapipe.shape[0])
    )

    # Concat
    print("----- Pós concat -----")
    combined_csv = pd.concat(
        [combined_csv_dlib, combined_csv_mediapipe], axis=0
    )
    print("Linhas, Colunas: " + str(combined_csv.shape))
    print(
        "Quantidade de Frames [Dlib]: "
        + str(combined_csv.loc[combined_csv.extractor == label_dlib].shape)
    )
    print(
        "Quantidade de Frames [Mediapipe]: "
        + str(
            combined_csv.loc[combined_csv.extractor == label_mediapipe].shape
        )
    )

    print("----- Description -----")
    mdnna = len(
        combined_csv.loc[
            (combined_csv.extractor == label_mediapipe)
            & (combined_csv["keys"].notna())
        ]
    ) / len(combined_csv)
    mdna = len(
        combined_csv.loc[
            (combined_csv.extractor == label_mediapipe)
            & (combined_csv["keys"].isna())
        ]
    ) / len(combined_csv)
    dbnna = len(
        combined_csv.loc[
            (combined_csv.extractor == label_dlib)
            & (combined_csv["keys"].notna())
        ]
    ) / len(combined_csv)
    dbna = len(
        combined_csv.loc[
            (combined_csv.extractor == label_dlib)
            & (combined_csv["keys"].isna())
        ]
    ) / len(combined_csv)

    print(
        "Porcentagem frames com landmarks detectados Mediapipe: {:.2f}%".format(
            mdnna
        )
    )
    print(
        "Porcentagem frames com landmarks NÃO detectados Mediapipe: {:.2f}%".format(
            mdna
        )
    )
    print(
        "Porcentagem frames com landmarks detectados Dlib: {:.2f}%".format(
            dbnna
        )
    )
    print(
        "Porcentagem frames com landmarks NÃO detectados Dlib: {:.2f}%".format(
            dbna
        )
    )

    combined_csv["na"] = np.where(combined_csv["keys"].isna(), 1, 0)

    return combined_csv


def plot_var_geral(df, extractors):
    # plot
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)

    colors = ["green", "red"]
    df1 = df.loc[df.extractor == extractors["extractor1"]]
    ax = sns.countplot(x="extractor", hue="na", data=df1, palette=colors)
    ax.set(ylabel="Total of Frames", title="Bar Count and Percent of Total")

    # add annotations
    for c in ax.containers:
        # custom label calculates percent and add an empty string so 0 value bars don't have a number
        labels = []

        for v in c:
            if v.get_height() > 0:
                elem = f"{v.get_height()/len(df1.index)*100:0.1f}%"
            else:
                elem = ""
            labels.append(elem)

        # labels = [f'{w/len(df1.index)*100:0.1f}%' if (w := v.get_height()) > 0 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type="edge")

    plt.subplot(2, 2, 2)
    df2 = df.loc[df.extractor == extractors["extractor2"]]
    ax = sns.countplot(x="extractor", hue="na", data=df2, palette=colors)
    ax.set(ylabel="Total of Frames", title="Bar Count and Percent of Total")

    # add annotations
    for c in ax.containers:
        # custom label calculates percent and add an empty string so 0 value bars don't have a number

        labels = [
            f"{v.get_height()/len(df2.index)*100:0.1f}%"
            if v.get_height() > 0
            else ""
            for v in c
        ]
        ax.bar_label(c, labels=labels, label_type="edge")


def create_directory(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")
    return dirName


def read_data_with_subdirectorys(data_path):
    videos_path_list = []
    print("List of all directories in '% s':" % data_path)

    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if name.endswith(".avi"):
                videos_path_list.append(os.path.join(path, name))

    return videos_path_list


def convert_gray_scale(input_path, video_output):

    files_avi = read_data_with_subdirectorys(input_path)

    for j in os.listdir(input_path):
        create_directory(str(video_output) + str(j))

    for i in files_avi:
        print(i)
        cap = cv2.VideoCapture(i)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        result = cv2.VideoWriter(
            video_output + i.split("UCF-101-Analysis/")[-1],
            cv2.VideoWriter_fourcc(*"MJPG"),
            25,
            size,
            isColor=False,
        )

        if cap.isOpened() == False:
            print("Error opening video stream or file")

        while cap.isOpened():
            _, image = cap.read()

            if _ == True:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                result.write(gray)
                # cv2.imshow(video_output, gray)
                # print(video_output+'Gray_'+i)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
            else:
                break

        cv2.destroyAllWindows()
        cap.release()

    return 1
