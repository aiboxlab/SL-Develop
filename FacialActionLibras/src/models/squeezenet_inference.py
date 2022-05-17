"""
----------> ATENÇÃO!!!!!!!!!!!!
ESTE SCRIPT FOI DESCONTINUADO,
PARA REALIZAR PREDIÇÕOS COM O SQUEEZENET USE squeezenet_inference_pp_experimental.py
E ESCOLHA O EXTRATOR DE FEATURES DESEJADO (mediapipe, dlib) 
O arquivo encontra-se nesse mesmo diretório!"""

import pickle as pkl
import xml.etree.ElementTree as etree
import xml.etree.ElementTree as et
from xml.dom import minidom
import os
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# Import necessary components for creation of xml file
from xml.etree import ElementTree, cElementTree
from xml.etree.ElementTree import tostring


"""from numba import jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer   
gpu_options.allow_growth = True"""


import cv2
import math
import dlib
import pandas as pd
import mediapipe as mp
import numpy as np
from config.config import (
    CASCADE_PATH,
    INDICESFACE,
    LABELS_L,
    LABELS_U,
    PREDICTOR_PATH,
    logger,
)
from keras import backend as K
from keras.models import load_model
from keras.utils import np_utils
from features.mediapipe.image_face_mesh_take import (
    get_landmarks_mediapipe_nomalized,
    predict_mediapipe_face_mesh,
    LANDMARK_IDS,
)

print(os.listdir())


# PREDICTOR_PATH = f'{ROOT_DIR}/models/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade = cv2.CascadeClassifier(CASCADE_PATH)
im_s = 96

lands_mp_match_dlib_682d = pd.read_csv('../data/examples/lands_mp_match_dlib_682d_ref02.csv', sep=';')

ScaleFactor = 1.15

def get_landmarks(im):
    logger.info("Getting the landmarks...")
    im = np.array(im, dtype="uint8")
    faces = cascade.detectMultiScale(im, ScaleFactor, 4, 0, (100, 100))#1.01 #mps-openface2

    if faces == ():
        print("NO get_landmarks FAAAAAAAAAAAAAAAAAAAAAAAAAACE DL")
        return np.matrix(
            [[0 for row in range(0, 2)] for col in range(INDICESFACE)]
        )
    else:
        print("SY get_landmarks FAAAAAAAAAAAAAAAAAAAAAAAAAACE DL")
        for (x, y, w, h) in faces:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            #for p in predictor(im, rect).parts():
                #print(p.x, p.y)
        matrix_ = np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
   
        return matrix_






def get_landmarks_mediapipe_selected(im):
    logger.info("Getting the landmarks...")
    
    im = np.array(im, dtype="uint8")
    mp_face_mesh = mp.solutions.face_mesh
    IMAGE_FILES = [im]
    height, width, _ = im.shape

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
    ) as face_mesh:
        for idx, file in enumerate(IMAGE_FILES):

            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                print("LAND NO FAAAAAAAAAAAAAAAAAAAAAAAAAACE MP")
                return np.matrix(
                    [[0 for row in range(0, 2)] for col in range(INDICESFACE)]
                )

            else:
                print("LAND SI FAAAAAAAAAAAAAAAAAAAAAAAAAACE MP")
            
                

                for face_landmarks in results.multi_face_landmarks:
                    x_y_lands = []
                    print(face_landmarks)
                    print(face_landmarks.shape)
                    for n in lands_mp_match_dlib_682d.landmark_mpipe:
                        
                        #substitua os valores x e y do dlib, pelos x e y preditos pelo mediapipe através do get no landmark correspondente
                        pt1 = face_landmarks.landmark[n-1]
                        x = int(pt1.x * width)
                        y = int(pt1.y * height)
                        
                        x_y_lands.append([x,y])
                        #print(x, y)
                matrix_ = np.matrix(x_y_lands)
                #vou colocar o return aqui para considerar apenas a primeira face
                return matrix_

def crop_face_mediapipe(im, i, image_path=None, save=False):
    if((image_path is not None) & (save is True)):
        img_write(f"{image_path}/img_video_frame_{i+1}.png", im)
    #print("###############")
    mp_face_mesh = mp.solutions.face_mesh
    #im = cv2.imread(im)
    IMAGE_FILES = [im]
    with mp_face_mesh.FaceMesh(
       min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    
        #image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_mesh.process(im)

        # Draw face detections of each face.
        if not results.multi_face_landmarks:#
            print("NO2 crop_face_mediapipe FACEEEEEEEEEEEEEEEEEEEEEEEEEE MP")
            l = np.matrix(
                [[0 for row in range(0, 2)] for col in range(INDICESFACE)]
            )
            rect = dlib.rectangle(0, 0, 0, 0)
            return np.zeros(im.shape, dtype=np.uint8), l
            
        else:
            annotated_image = im.copy()
                
            for face_landmarks in results.multi_face_landmarks:#
                print("SY2 crop_face_mediapipe FACEEEEEEEEEEEEEEEEEEEEEEEEEE MP")

                h, w, c = im.shape
                cx_min = w
                cy_min = h
                cx_max = cy_max = 0
                for id, lm in enumerate(face_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    #cx = min(math.floor(lm.x * w), w - 1)
                    #cy = min(math.floor(lm.y * h), h - 1)
                    if cx < cx_min:
                        cx_min = cx
                    if cy < cy_min:
                        cy_min = cy
                    if cx > cx_max:
                        cx_max = cx
                    if cy > cy_max:
                        cy_max = cy

                cx_min = cx_min-1
                cy_min = cy_min-1
                cx_max = cx_max+1
                cy_max = cy_max+1
                
                #desenhe o retangle
                #annotated_image = cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)

                
                sub_face = im[cy_min : cy_max, cx_min : cx_max]

            return sub_face, None
                

def crop_face_dlib(im, i, image_path=None, save=False):
    if((image_path is not None) & (save is True)):
        img_write(f"{image_path}/img_video_frame_{i+1}.png", im)

    faces = cascade.detectMultiScale(im, ScaleFactor, 4, 0, (100, 100))
    if faces == ():
        print("NO crop_face_dlib FAAAAAAAAAAAAAAAAAAAAAAAAAACE DL")
        l = np.matrix(
            [[0 for row in range(0, 2)] for col in range(INDICESFACE)]
        )
        rect = dlib.rectangle(0, 0, 0, 0)
        return np.zeros(im.shape, dtype=np.uint8), l

        
    else:
        print("SY crop_face_dlib FAAAAAAAAAAAAAAAAAAAAAAAAAACE DL")
        for (x, y, w, h) in faces:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            l = np.array(
                [[p.x, p.y] for p in predictor(im, rect).parts()],
                dtype=np.float32,
            )
            sub_face = im[y : y + h, x : x + w]
        return sub_face, l


def img_write(path, img):
    cv2.imwrite(path, img)


def annotate_landmarks(im, landmarks):
    img = im.copy()
    if landmarks.all() == 0:
        return im
    else:
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(img, pos, 2, color=(255, 255, 255), thickness=-1)
        return img


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError("The lowest choosable beta is zero (only precision).")
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)


def load_h5_model(path):
    logger.info("Loading the H5 model")
    model = load_model(
        path,
        custom_objects={
            "fmeasure": fmeasure,
            "precision": precision,
            "recall": recall,
        },
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", fmeasure, precision, recall],
    )
    return model


def load_le(path):
    res = []
    with open(path, "rb") as file:
        res = pkl.load(file)
    return res


def load_y_u():
    logger.info("Loading the upper train2 file")
    x_u_train2 = np.load(f"../data/annotations/x_u_train2.npy")

    logger.info("Loading the y_u file")
    y_u = np.load(f"../data/annotations/y_u.npy")
    y_u_train1 = np.load(f"../data/annotations/y_u_corpus.npy")
    y_u_train2 = np.load(f"../data/annotations/y_u_train2.npy")

    logger.info("Making the Y_u")
    Y_u = np.append(
        np.append(y_u, y_u_train1),
        y_u_train2[: int(x_u_train2.size / (60 * 97 * 1))],
    )
    return Y_u


def load_y_l():
    logger.info("Loading the x_l_train2 file")
    x_l_train2 = np.load(f"../data/annotations/x_l_train2.npy")

    logger.info("Loading the y_l_train2 file")
    y_l = np.load(f"../data/annotations/y_l.npy")
    y_l_train1 = np.load(f"../data/annotations/y_l_corpus.npy")
    y_l_train2 = np.load(f"../data/annotations/y_l_train2.npy")

    logger.info("Loading the Y_l file")
    Y_l = np.append(
        np.append(y_l, y_l_train1),
        y_l_train2[: int(x_l_train2.size / (36 * 98 * 1))],
    )
    Y_l = np.nan_to_num(Y_l)
    return Y_l


def ordered_u(encoded_Y_u):
    Y_u = np_utils.to_categorical(encoded_Y_u)
    labels_encoded_u = encoder_u.inverse_transform(encoded_Y_u)
    labels_ordered_u = np.sort(labels_encoded_u)
    labels_ordered_u = np.append(labels_ordered_u, 74)
    labels_ordered_u = set(labels_ordered_u)
    labels_ordered_u = np.fromiter(
        labels_ordered_u, int, len(labels_ordered_u)
    )
    return labels_ordered_u


def ordered_l(encoded_Y_l):
    Y_l = np_utils.to_categorical(encoded_Y_l)
    num_classes_l = Y_l.shape[1]
    labels_encoded_l = encoder_l.inverse_transform(encoded_Y_l)
    labels_ordered_l = np.sort(labels_encoded_l)
    labels_ordered_l = np.append(labels_ordered_l, 73)
    labels_ordered_l = set(labels_ordered_l)
    labels_ordered_l = np.fromiter(
        labels_ordered_l, int, len(labels_ordered_l)
    )
    return labels_ordered_l


## Loading models
modell = load_h5_model("../models/squeezenet_l_corpus_6.h5")
modelu = load_h5_model("../models/squeezenet_u_corpus_6.h5")

encoder_u = load_le("../models/y_u_encoder.pkl")
encoder_l = load_le("../models/y_l_encoder.pkl")

labels_ordered_u = ordered_u(encoder_u.transform(load_y_u()))
labels_ordered_l = ordered_l(encoder_l.transform(load_y_l()))
## Loading models


def _get_landmarks_mediapipe(img_np):
    landmarks = predict_mediapipe_face_mesh(img_np)
    res_media = get_landmarks_mediapipe(img_np, landmarks)
    #print(img_np)
    print(res_media)
    res_media = list(map(lambda x: list(res_media[x]), LANDMARK_IDS))
    res_media = np.asanyarray(res_media)
    return res_media


def neural_net(path, name_video, df_results, image_path=None, save=False, extractor=None):
    logger.info("Making predictions...")
    v_entry = cv2.VideoCapture(path, 0)
    frames = int(v_entry.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = v_entry.get(cv2.CAP_PROP_FPS)
    print("---------------------------------------------------------------------")
    print(fps)
    
    logger.info(f"FPS: {fps}")
    output_u = []
    output_l = []
    M = []
    xy = []
        
    for i, j in enumerate(range(0, frames)):
        
        print("--------------------------")
        print(f"Frame[{j+1}]")
        print(image_path)

        v_entry.set(1, int(j))
        ret, im = v_entry.read()
        if ret is True:

            if(extractor == 'mediapipe'):
                a, c = crop_face_mediapipe(im, i, image_path, save)
                #c = get_landmarks_mediapipe_selected(a)
            elif(extractor == 'dlib'):
                #a, l = crop_face_dlib(im, i, image_path, save)
                a = im
                c = get_landmarks(a)
            elif(extractor == 'mediapipeVMaan'):
                a, l = crop_face_mediapipe(im, i, image_path, save)
                #a = im
                c = _get_landmarks_mediapipe(a)


            #print(STOP)
            #c = get_landmarks_mediapipe_selected(a) #Exec 1º #1min17sec #18:54
            #print(c)

            xy.append(c)

            print(c)

            points_u[:9, :] = c[17:26, :]
            points_u[10:, :] = c[36:47, :]
            vp = np.stack((points_u))
            points_l[:12, :] = c[2:14, :]
            points_l[13:, :] = c[48:67, :]
            vb = np.stack((points_l))
            vs_brown_e = np.squeeze(np.asarray(c[19] - c[17]))
            vi_brown_e = np.squeeze(np.asarray(c[21] - c[17]))
            vs_brown_d = np.squeeze(np.asarray(c[24] - c[26]))
            vi_brown_d = np.squeeze(np.asarray(c[22] - c[26]))
            a_brown_e = np.arccos(
                np.dot(vs_brown_e, vi_brown_e, out=None)
                / (np.linalg.norm(vs_brown_e) * np.linalg.norm(vi_brown_e))
            )
            a_brown_d = np.arccos(
                np.dot(vs_brown_d, vi_brown_d, out=None)
                / (np.linalg.norm(vs_brown_d) * np.linalg.norm(vi_brown_d))
            )
            v1_eye_e = np.squeeze(np.asarray(c[37] - c[41]))
            v2_eye_e = np.squeeze(np.asarray(c[38] - c[40]))
            v1_eye_d = np.squeeze(np.asarray(c[43] - c[47]))
            v2_eye_d = np.squeeze(np.asarray(c[44] - c[46]))
            vs = np.stack(
                (
                    vs_brown_e,
                    vi_brown_e,
                    vs_brown_d,
                    vi_brown_d,
                    v1_eye_e,
                    v2_eye_e,
                    v1_eye_d,
                    v2_eye_d,
                )
            )
            d_lips_h1 = np.squeeze(np.asarray(c[48] - c[54]))
            d_lips_h2 = np.squeeze(np.asarray(c[60] - c[64]))
            d_lips_v1 = np.squeeze(np.asarray(c[51] - c[57]))
            d_lips_v2 = np.squeeze(np.asarray(c[62] - c[66]))
            vl = np.stack((d_lips_h1, d_lips_h2, d_lips_v1, d_lips_v2))
            p_u = [vp.tolist(), vs.tolist()]
            points_upper = np.hstack(
                [np.hstack(np.vstack(p_u)), a_brown_e, a_brown_d]
            )
            p_l = [vb.tolist(), vl.tolist()]
            points_lower = np.hstack(np.vstack(p_l)).reshape((36, 2))
            r = cv2.resize(
                a, dsize=(im_s, im_s), interpolation=cv2.INTER_CUBIC
            )
            r = r[:, :, 1]
            upper = np.array(r[:60, :])
            lower = np.array(r[60:, :])
            im_u = np.vstack((upper.T, points_upper))
            im_u = im_u.astype("float32")
            im_u /= 255
            im_l = np.vstack((lower.T, points_lower[:, 0], points_lower[:, 1]))
            im_l = im_l.astype("float32")
            im_l /= 255
            x_upper = np.expand_dims(im_u, axis=0)
            x_lower = np.expand_dims(im_l, axis=0)
            x_upper = x_upper.reshape((1, 60, 97, 1))
            x_lower = x_lower.reshape((1, 36, 98, 1))
            exit_u = modelu.predict(x_upper)
            exit_l = modell.predict(x_lower)

            exit_u = np.argmax(exit_u, axis=1)
            exit_l = np.argmax(exit_l, axis=1)
            e_labels_u = encoder_u.inverse_transform(exit_u)
            e_labels_l = encoder_l.inverse_transform(exit_l)
            # logger.info(e_labels_u)
            # logger.info(e_labels_l)
            output_u = np.append(output_u, e_labels_u)
            output_l = np.append(output_l, e_labels_l)
        else:
            output_u = np.append(output_u, 74)
            output_l = np.append(output_l, 72)
            continue

    all_exit_u = np.matrix(zip(range(0, frames), output_u))
    all_exit_l = np.matrix(zip(range(0, frames), output_l))

    root = et.Element(
        "TIERS",
        **{"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance"},
        **{"xsi:noNamespaceSchemaLocation": "file:avatech-tiers.xsd"},
    )

    """df_results = pd.DataFrame(
            columns=["name_video", "end", "frame", "start", 'aus']
        )"""


    somedata = et.SubElement(root, "TIER", columns="AUs")
    for m, n in enumerate(range(0, frames)):
        print("--------------------------")
        print(f"Frame[{n+1}]")
        if np.where(labels_ordered_u == output_u[m]):
            a = np.where(labels_ordered_u == output_u[m])
            logger.info(a)
            print(
                f"Label for Upper Face at frame[{n+1}]: {LABELS_U[int(a[0][0])]}"
            )
            if np.where(labels_ordered_l == output_l[m]):
                b = np.where(labels_ordered_l == output_l[m])
                logger.info(b)
                print(
                    f"Label for Lower Face at frame[{n+1}]: {LABELS_L[int(b[0][0])]}"
                )
                ms_inicial = round((m * (1000 / (fps / 1.001))) * 0.001, 3)
                ms_final = round(((m + 1) * (1000 / (fps / 1.001))) * 0.001, 3)
                child1 = ElementTree.SubElement(
                    somedata,
                    "span",
                    start="%s" % (ms_inicial),
                    end="%s" % (ms_final),
                    frame="%s" % (n + 1),
                )
                v = etree.Element("v")
                v.text = "%s+%s" % (
                    LABELS_U[int(a[0][0])],
                    LABELS_L[int(b[0][0])],
                )
                child1.append(v)
                tree = cElementTree.ElementTree(
                    root
                )  # wrap it in an ElementTree instance, and save as XML
                t = minidom.parseString(
                    ElementTree.tostring(root)
                ).toprettyxml()  # Since ElementTree write() has no pretty printing support, used minidom to beautify the xml.
                tree1 = ElementTree.ElementTree(ElementTree.fromstring(t))
                logger.info(tree1)

                new_row = {
                            "video_name": name_video,
                            "end": str(ms_final),
                            "frame": str((n + 1)),
                            "start": str(ms_inicial),
                            "aus": v.text,
                            "xy": xy[n].tolist()
                        }

                df_results = df_results.append(new_row, ignore_index=True)

            else:
                continue
        else:
            continue

    return tree1, df_results

