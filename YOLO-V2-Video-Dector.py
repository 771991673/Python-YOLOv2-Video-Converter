# For more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
import numpy as np
import cv2
import os

COLOR_RGB = (255, 102, 255) # Text and Rectangle color : Purple

FILE_INPUT = 'input.mp4'
FILE_OUTPUT = 'output.avi'

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

# Playing video from file:
# cap = cv2.VideoCapture('vtest.avi')
# Capturing video from webcam:
cap = cv2.VideoCapture(FILE_INPUT)
FPS_INPUT = cap.get(cv2.CAP_PROP_FPS)

currentFrame = 0

width = cap.get(3)
height = cap.get(4)


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(FILE_OUTPUT,fourcc, FPS_INPUT, (int(width),int(height)))


# Minimum confidence threshold. Increasing this will improve false positives but will also reduce detection rate.
min_confidence=0.14
model = 'yolov2.weights'
config = 'yolov2.cfg'


#Load names of classes
classes = None
with open('labels.txt', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
print(classes)

# Load weights and construct graph
net = cv2.dnn.readNetFromDarknet(config, model)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

winName = 'Running YOLO Model'
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:
        # Handles the mirroring of the current frame
        # frame = cv2.flip(frame,1)

        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (416, 416), True, crop=False)
        net.setInput(blob)
        predictions = net.forward()
        probability_index = 5

        for i in range(predictions.shape[0]):
            prob_arr = predictions[i][probability_index:]
            class_index = prob_arr.argmax(axis=0)
            confidence = prob_arr[class_index]
            # x_center, y_center: center of box, width_box, height_box: width, height of box
            if confidence > min_confidence:
                x_center = predictions[i][0] * width
                y_center = predictions[i][1] * height
                width_box = predictions[i][2] * width
                height_box = predictions[i][3] * height

                x1 = int(x_center - width_box * 0.5)
                y1 = int(y_center - height_box * 0.5)
                x2 = int(x_center + width_box * 0.5)
                y2 = int(y_center + height_box * 0.5)

                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RGB, 3)
                cv2.putText(frame, classes[class_index] + " " + "{0:.1f}".format(confidence), (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RGB, 2, cv2.LINE_AA)


        # Saves for video
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('frame',frame)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
