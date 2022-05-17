import cv2

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter(
    "jayne-webcam.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, size
)


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    result.write(image)
    cv2.imshow("MediaPipe FaceMesh", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
result.release()
cv2.destroyAllWindows()
