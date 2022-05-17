import glob
import os
import pickle as pkl

import numpy as np
from config.config import logger
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import os
import time


def tic():
    _start_time = time.time()
    return _start_time


def tac(_start_time):
    # Reference: https://newbedev.com/how-do-you-determine-a-processing-time-in-python
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    return (
        time.time(),
        "Time passed: {}hour:{}min:{}sec".format(t_hour, t_min, t_sec),
        t_hour, t_min, t_sec
    )


def create_directory_v2(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")
    return dirName


def read_data_with_subdirectorys(data_path, extension):
    videos_path_list = []
    print("List of all directories in '% s':" % data_path)

    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if(extension is not None):
                if name.endswith(extension):
                    videos_path_list.append(os.path.join(path, name))
            else:
                videos_path_list.append(os.path.join(path, name))

    return videos_path_list


def get_all_files_from_directory(path):
    return glob.glob(path)


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_AUs_info(xml_object):
    for element in xml_object.getiterator():
        dict_keys = {}
        if element.keys():
            for name, value in element.items():
                dict_keys[name] = value
            print(dict_keys)


# x_u = np.load('x_u.npy')
# x_u_train1 = np.load('.../Data_annotations/x_u_corpus.npy')
# x_u_train2 = np.load('./Data_annotations/x_u_train2.npy')
logger.info("Loading the upper train2 file")
x_u_train2 = np.load(f"../data/annotations/x_u_train2.npy")
#
# y_u = np.load('./Data_annotations/y_u.npy')
# y_u_train1 = np.load('./Data_annotations/y_u_corpus.npy')
# y_u_train2 = np.load('./Data_annotations/y_u_train2.npy')

logger.info("Loading the y_u file")
y_u = np.load(f"../data/annotations/y_u.npy")
y_u_train1 = np.load(f"../data/annotations/y_u_corpus.npy")
y_u_train2 = np.load(f"../data/annotations/y_u_train2.npy")
##############################################################################
##############################################################################
# X_u=np.append(np.append(x_u,x_u_train1),x_u_train2)
logger.info("Making the Y_u")
Y_u = np.append(
    np.append(y_u, y_u_train1),
    y_u_train2[: int(x_u_train2.size / (60 * 97 * 1))],
)
# X_u=np.nan_to_num(X_u)
# X_u = X_u.astype('float32')
##############################################################################
##############################################################################
img_rows_u, img_cols_u = 60, 97
# X_u = X_u.reshape(int(X_u.shape[0]/(60*97*1)), img_rows_u, img_cols_u, 1)
input_shape = (img_rows_u, img_cols_u, 1)
##############################################################################
##############################################################################
# convert class vectors to binary class matrices
logger.info("Making the LabelEncoder")
encoder_u = LabelEncoder()
encoder_u.fit(Y_u)
encoded_Y_u = encoder_u.transform(Y_u)

# convert integers to dummy variables (i.e. one hot encoded)
Y_u = np_utils.to_categorical(encoded_Y_u)
num_classes_u = Y_u.shape[1]
labels_encoded_u = encoder_u.inverse_transform(encoded_Y_u)
labels_ordered_u = np.sort(labels_encoded_u)
labels_ordered_u = np.append(labels_ordered_u, 74)
labels_ordered_u = set(labels_ordered_u)
labels_ordered_u = np.fromiter(labels_ordered_u, int, len(labels_ordered_u))
##############################################################################
##############################################################################
# x_l = np.load("x_l.npy")
# x_l_train1 = np.load(".../Data_annotations/x_l_corpus.npy")
logger.info("Loading the x_l_train2 file")
x_l_train2 = np.load(f"../data/annotations/x_l_train2.npy")
#
logger.info("Loading the y_l_train2 file")
y_l = np.load(f"../data/annotations/y_l.npy")
y_l_train1 = np.load(f"../data/annotations/y_l_corpus.npy")
y_l_train2 = np.load(f"../data/annotations/y_l_train2.npy")
##############################################################################
##############################################################################
# X_l=np.append(np.append(x_l,x_l_train1),x_l_train2)
logger.info("Loading the Y_l file")
Y_l = np.append(
    np.append(y_l, y_l_train1),
    y_l_train2[: int(x_l_train2.size / (36 * 98 * 1))],
)
# X_l=np.nan_to_num(X_l)
Y_l = np.nan_to_num(Y_l)
# X_l = X_l.astype('float32')
##############################################################################
##############################################################################
img_rows_l, img_cols_l = 36, 98
# X_l = X_l.reshape(int(X_l.shape[0]/(36*98*1)), img_rows_l, img_cols_l, 1)
input_shape = (img_rows_l, img_cols_l, 1)
##############################################################################
##############################################################################
# convert class vectors to binary class matrices
logger.info("Making the encoder_l")
encoder_l = LabelEncoder()
encoder_l.fit(Y_l)
encoded_Y_l = encoder_l.transform(Y_l)

with open("../models/y_l_encoder.pkl", "wb") as file:
    pkl.dump(encoder_l, file, protocol=pkl.HIGHEST_PROTOCOL)

# convert integers to dummy variables (i.e. one hot encoded)
Y_l = np_utils.to_categorical(encoded_Y_l)
num_classes_l = Y_l.shape[1]
labels_encoded_l = encoder_l.inverse_transform(encoded_Y_l)
labels_ordered_l = np.sort(labels_encoded_l)
labels_ordered_l = np.append(labels_ordered_l, 73)
labels_ordered_l = set(labels_ordered_l)
labels_ordered_l = np.fromiter(labels_ordered_l, int, len(labels_ordered_l))
# print(labels_ordered_l)
##############################################################################
##############################################################################
