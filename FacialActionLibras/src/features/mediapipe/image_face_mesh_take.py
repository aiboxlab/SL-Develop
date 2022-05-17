import mediapipe as mp
import dataclasses
import math
import numpy as np
import cv2
import pandas as pd
from protobuf_to_dict import protobuf_to_dict

from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import landmark_pb2
from typing import List, Mapping, Optional, Tuple, Union

## Carregando IDs de mapeamento entre mediapipe e dlib
lands_mp_match_dlib_682d = pd.read_csv('../data/examples/lands_mp_match_dlib_682d_ref02.csv', sep=';')

## Atribuindo IDs de mapeamento entre mediapipe e dlib
LANDMARK_IDS = lands_mp_match_dlib_682d.landmark_mpipe.to_list()

mp_face_mesh = mp.solutions.face_mesh
# Load drawing_utils and drawing_styles
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

MEDIAPIPE_LANDMARKS = 468

from config.config import (
    INDICESFACE
)

@dataclasses.dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = WHITE_COLOR
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (
            value < 1 or math.isclose(1, value)
        )

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)

    return x_px, y_px


def get_landmarks_mediapipe_nomalized(
    image: np.ndarray, landmark_list: landmark_pb2.NormalizedLandmarkList, autoencoder=False
):
    landmarks = np.matrix(
            [[0 for row in range(0, 2)] for col in range(INDICESFACE)]
        )

    if(not landmark_list):
        # print("Vazio")#mps
        return landmarks
    
    keypoints = protobuf_to_dict(landmark_list)
    #print(keypoints)

    if(len(keypoints["landmark"]) == MEDIAPIPE_LANDMARKS):
        #print("if len(keypoints.keys()) == MEDIAPIPE_LANDMARKS): {} == {}".format(len(keypoints["landmark"]), MEDIAPIPE_LANDMARKS))
        
        image_rows, image_cols, _ = image.shape
        idx_to_coordinates = {}
        count_ids = 0

        for idx, landmark in enumerate(landmark_list.landmark):
            landmark_px = _normalized_to_pixel_coordinates(
                landmark.x, landmark.y, image_cols, image_rows
            )
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px

                ##Verifica se os ids que queremos estão presentes
                if idx in LANDMARK_IDS:
                    count_ids+=1
                
        landmarks = idx_to_coordinates
        # print("Count Ids:", count_ids, "Landmarks:", len(landmarks))

        #Se mesmo após normalizar, algum landmarks ficar fora do quadro de pixel...
        # if(len(landmarks)==MEDIAPIPE_LANDMARKS):
        if(not autoencoder and count_ids==INDICESFACE):
            #print("if(len(landmarks)==MEDIAPIPE_LANDMARKS): {} == {}".format(str(len(landmarks)), MEDIAPIPE_LANDMARKS))

            pass
        elif (autoencoder and len(landmarks) == 468):
            pass
        else:
            #print("else do if(len(landmarks)==MEDIAPIPE_LANDMARKS): {} == {}".format(str(len(landmarks)), MEDIAPIPE_LANDMARKS))
            landmarks = np.matrix(
                [[0 for row in range(0, 2)] for col in range(INDICESFACE)]
            )

    else:
        #print("else do len(keypoints.keys()) == MEDIAPIPE_LANDMARKS): {} == {}".format(len(keypoints["landmark"]), MEDIAPIPE_LANDMARKS))
        return landmarks

    

    #print("landmarkssssssssssssssssssssssssssss")
    #print(landmarks)
    return landmarks


    


def predict_mediapipe_face_mesh(image_np):
    # Run MediaPipe Face Mesh.
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        #refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return

        for face_landmarks in results.multi_face_landmarks:
            return face_landmarks
