import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#from mp.framework.formats import detection_pb2
#from mp.framework.formats import location_data_pb2


import math
from typing import List, Mapping, Optional, Tuple, Union
import dataclasses

import cv2
import dlib
import pandas as pd
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.utils import np_utils

INDICESFACE = 68
PREDICTOR_PATH = "../../models/shape_predictor_68_face_landmarks.dat"
CASCADE_PATH = "../../models/haarcascade_frontalface_default.xml"


# PREDICTOR_PATH = f'{ROOT_DIR}/models/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
# cascade_path=f'{ROOT_DIR}/models/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(CASCADE_PATH)
im_s = 96
# For static images:
mp_face_mesh = mp.solutions.face_mesh

_RGB_CHANNELS = 3


WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

@dataclasses.dataclass
class DrawingSpec:
  # Color for drawing the annotation. Default to the white color.
  color: Tuple[int, int, int] = WHITE_COLOR
  # Thickness for drawing the annotation. Default to 2 pixels.
  thickness: int = 2
  # Circle radius. Default to 2 pixels.
  circle_radius: int = 2


def img_write(path, img):
    cv2.imwrite(path, img)


def crop_face_dlib(im,image_name):
    #if((image_path is not None) & (save is True)):
        #img_write(f"{image_path}/app2_img_video_frame_{i+1}.png", im)
    im = cv2.imread(im)

    #https://stackoverflow.com/questions/20801015/recommended-values-for-opencv-detectmultiscale-parameters
    #https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
    #gray, scaleFactor, minNeighbors, minSize (Tamanho mínimo possível do objeto. Objetos menores que isso são ignorados), flags
    faces = cascade.detectMultiScale(im, 1.15, 4, 0, (100, 100))
    if(faces == ()):
        print("NO FACEEEEEEEEEEEEEEEEEEEEEEEEEE")
        l = np.matrix(
            [[0 for row in range(0, 2)] for col in range(INDICESFACE)]
        )
        rect = dlib.rectangle(0, 0, 0, 0)
        return np.zeros(im.shape, dtype=np.uint8), l
    else:
        print("SY FACEEEEEEEEEEEEEEEEEEEEEEEEEE")
        #Pelo que entendi, na iteração abaixo é retornada a última face apenas, pos as demais são sobrescritas
        for (x, y, w, h) in faces:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            l = np.array(
                [[p.x, p.y] for p in predictor(im, rect).parts()],
                dtype=np.float32,
            )
            sub_face = im[y : y + h, x : x + w]
            img_write("../../data/examples/{0}}_crop_dlib.png".format(image_name), sub_face)
            print(l.shape)
        return sub_face, l


def _normalized_to_pixel_coordinates(
    normalized_x: float, 
    normalized_y: float, 
    image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                        math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    print("x_px, y_px")
    print(x_px, y_px)
    return x_px, y_px





"""def crop_face_mp_face_detector(im, image_name):

    IMAGE_FILES = [im]
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:

        
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            if not results.detections:
                print("NO FACEEEEEEEEEEEEEEEEEEEEEEEEEE")
                continue
            else:
                annotated_image = image.copy()
                for detector in results.detections:

                    mp_drawing.draw_detection(annotated_image, detector)
                    #eu_draw_detection(annotated_image, detection)

                    cv2.imshow('window', annotated_image)
                    cv2.waitKey(0)
                    cv2.imwrite('../../data/examples/{0}annotated_image_detector.png'.format(image_name), annotated_image)"""


def crop_face_mp_minmax(im, image_name):

    IMAGE_FILES = [im]
    with mp_face_mesh.FaceMesh(
       min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        
        for idx, file in enumerate(IMAGE_FILES):
            print(file)
            image = cv2.imread(file)
            cv2.imshow('window', image)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_mesh.process(image)

            # Draw face detections of each face.
            if not results.multi_face_landmarks:#
                print("NO FACEEEEEEEEEEEEEEEEEEEEEEEEEE")
                continue
            else:
                annotated_image = image.copy()
                    
                for face_landmarks in results.multi_face_landmarks:#
                    print("SY FACEEEEEEEEEEEEEEEEEEEEEEEEEE")

                    h, w, c = image.shape
                    cx_min = w
                    cy_min = h
                    cx_max = cy_max = 0
                    for id, lm in enumerate(face_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if cx < cx_min:
                            cx_min = cx
                        if cy < cy_min:
                            cy_min = cy
                        if cx > cx_max:
                            cx_max = cx
                        if cy > cy_max:
                            cy_max = cy
                    
                    #desenhe o retangle
                    #annotated_image = cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)

                   
                    sub_face = image[cy_min : cy_max, cx_min : cx_max]
                    
                    #print(detection)
                    #mp_drawing.draw_detection(annotated_image, face_landmarks)
                    #eu_draw_detection(annotated_image, detection)

                    cv2.imshow('window', sub_face)
                    cv2.waitKey(0)
                    cv2.imwrite('../../data/examples/{0}_annotated_image_detector.png'.format(image_name), sub_face)



def crop_face_mp_face_mesh(im, image_name):

    IMAGE_FILES = [im]
    with mp_face_mesh.FaceMesh(
       min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_mesh.process(image)

            # Draw face detections of each face.
            if not results.multi_face_landmarks:#
                print("NO FACEEEEEEEEEEEEEEEEEEEEEEEEEE")
                continue
            else:
                annotated_image = image.copy()

                for face_landmarks in results.multi_face_landmarks:
                    """mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )"""
                    height, width, _ = annotated_image.shape
                    for n in range(0, 468):
                        pt1 = face_landmarks.landmark[n]
                        x = int(pt1.x * width)
                        y = int(pt1.y * height)

                        annotated_image = cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

                    
                    cv2.imshow('window', annotated_image)
                    cv2.waitKey(0)
                    cv2.imwrite('../../data/examples/{0}_annotated_image_detector.png'.format(image_name), annotated_image)



def crop_face_mediapipe_proporcional(im, image_name, factor):

    IMAGE_FILES = [im]
    with mp_face_mesh.FaceMesh(
       min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        
        for idx, file in enumerate(IMAGE_FILES):
            print(file)
            image = cv2.imread(file)
            cv2.imshow('window', image)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_mesh.process(image)

            # Draw face detections of each face.
            if not results.multi_face_landmarks:#
                print("NO FACEEEEEEEEEEEEEEEEEEEEEEEEEE")
                continue
            else:
                annotated_image = image.copy()
                    
                for face_landmarks in results.multi_face_landmarks:#
                    print("SY FACEEEEEEEEEEEEEEEEEEEEEEEEEE")

                    h, w, c = image.shape
                    cx_min = w
                    cy_min = h
                    cx_max = cy_max = 0
                    for id, lm in enumerate(face_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if cx < cx_min:
                            cx_min = cx
                        if cy < cy_min:
                            cy_min = cy
                        if cx > cx_max:
                            cx_max = cx
                        if cy > cy_max:
                            cy_max = cy
                    
                    #desenhe o retangle
                    #annotated_image = cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)


                    xfactor = math.ceil((cx_max-cx_min)*factor)
                    yfactor = math.ceil((cy_max-cy_min)*factor)

                    xfactor_ = math.ceil((cx_max+cx_min)*factor)
                    yfactor_ = math.ceil((cy_max+cy_min)*factor)

                    """cx_min = cx_min-xfactor
                    cy_min = cy_min-yfactor
                    cx_max = cx_max+xfactor
                    cy_max = cy_max+yfactor"""


                    sub_face = image[cy_min : cy_max, cx_min : cx_max]
                    
                    #print(detection)
                    #mp_drawing.draw_detection(annotated_image, face_landmarks)
                    #eu_draw_detection(annotated_image, detection)

                    cv2.imshow('window', image)
                    cv2.waitKey(0)
                    #cv2.imwrite('../../data/examples/{0}_annotated_image_detector.png'.format(image_name), sub_face)


                #desenhe o retangle
                image = cv2.rectangle(image, (cx_min-xfactor, cy_min-yfactor), (cx_max+xfactor, cy_max+yfactor), (255, 0, 0), 2) #-- Proporção blue
                image = cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (0, 255, 255), 2) #original amarelo
                image = cv2.rectangle(image, (cx_min-xfactor_, cy_min-yfactor_), (cx_max+xfactor_, cy_max+yfactor_), (255, 255, 0), 2) #++ azul claro
                cv2.imshow('window', image)
                cv2.waitKey(0)

if __name__ == "__main__":
    image_name = 'download'
    ext = 'png'
    #image_name = "img_video_frame_1.png"
    #image_name = "app_img_video_frame_4.png" #Dlib e MP
    image = "../../data/examples/{0}.{1}".format(image_name, ext)
    factor = 0.25

    #crop_face(image, image_name)
    crop_face_mediapipe_proporcional(image, image_name, factor)

