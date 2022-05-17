import cv2
import os
 
# Opens the inbuilt camera of laptop to capture video.
cap = cv2.VideoCapture(0)
i = 0

video_name = 'RightVideoSN008_comp'
video_input = "../../data/examples/{}.avi".format(video_name)
cap = cv2.VideoCapture(video_input)
ret, frame = cap.read()
h, w, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc(*"XVID")
print("jglksdfjgklsfjklsfjgbh")


def create_directory(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")
    return dirName

create_directory("../../data/examples/{}-frames/".format(video_name))

while(cap.isOpened()):
    ret, frame = cap.read()
     
    # This condition prevents from infinite looping
    # incase video ends.
    if ret == False:
        break
     
    # Save Frame by Frame into disk using imwrite method
    print(i)
    cv2.imwrite('../../data/examples/'+video_name+'-frames/Frame'+str(i)+'.jpg', frame)
    i += 1
 
cap.release()
cv2.destroyAllWindows()
