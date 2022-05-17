import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from protobuf_to_dict import protobuf_to_dict

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

video_output = "../../../data/processed/examples/"
df_keys = pd.DataFrame(columns=["frame", "video_name", "keys"])


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter(
    str(video_output) + "mediapipe-jayne-webcam.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    30,
    size,
)
count = 0
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    #refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        print(count)
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

                keypoints = protobuf_to_dict(face_landmarks)
                # arr_external = np.array([[0,0,0]])

                arr_external = np.array(
                    [
                        [
                            keypoints["landmark"][0]["x"],
                            keypoints["landmark"][0]["y"],
                            keypoints["landmark"][0]["z"],
                        ]
                    ]
                )

                for i in range(1, len(keypoints["landmark"])):
                    arr_internal_ = np.array(
                        [
                            [
                                keypoints["landmark"][i]["x"],
                                keypoints["landmark"][i]["y"],
                                keypoints["landmark"][i]["z"],
                            ]
                        ]
                    )

                    arr_external = np.concatenate(
                        (arr_external, arr_internal_), axis=0
                    )

                new_row = {
                    "frame": int(result.get(cv2.CAP_PROP_POS_FRAMES)),
                    "video_name": "teste_webcam_write",
                    "keys": arr_external,
                }
        else:
            print("NOOOT")
            arr_external = np.array([[0, 0, 0]])
            new_row = {
                "frame": int(result.get(cv2.CAP_PROP_POS_FRAMES)),
                "video_name": "teste_webcam_write",
                "keys": None,
            }
        count = count + 1
        print("Frame " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        print("Qt Keys " + str(len(keypoints["landmark"])))

        result.write(image)
        cv2.imshow("MediaPipe FaceMesh", image)

        print(arr_external)
        print("------------------------")

        df_keys = df_keys.append(new_row, ignore_index=True)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

df_keys.to_csv(
    str(video_output) + str("teste_webcam_write") + str(".csv"), sep=";"
)

cap.release()
result.release()
cv2.destroyAllWindows()
