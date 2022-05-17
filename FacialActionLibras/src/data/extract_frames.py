import os
import cv2 as cv

augmentations = [
    "adjust_brightness",
    "adjust_jpeg_quality",
    "rgb_to_grayscale",
    "rot90",
]
videos = [
    "RostoIntensidade-07Setima-Zangado",
    "RostoIntensidade-04Quarta-Surpresa",
    "RostoIntensidade-07Setima-Sortudo",
    "RostoIntensidade-05Quinta-Acalmar",
]

input_base_path = "../../data/processed/tamires_augmentation/"
output_base_path = "../../data/examples/frames_tamires_augmented/"

for aug in augmentations:
    for vid in videos:
        print(aug, "+", vid)

        # Creates the folders structure
        if os.path.isdir(output_base_path + aug + "/" + vid):
            pass
            # print("Folder already created")
        else:
            try:
                os.makedirs(output_base_path + aug + "/" + vid)
            except OSError as error:
                print(error)

        # Captures the frames
        count = 1
        vidcap = cv.VideoCapture(input_base_path + aug + "/" + vid + ".avi")
        success, image = vidcap.read()

        while success:
            cv.imwrite(
                output_base_path
                + aug
                + "/"
                + vid
                + "/img_video_frame_%d.png" % count,
                image,
            )
            success, image = vidcap.read()
            print("Read a new frame: ", count, success)
            count += 1

        vidcap.release()
