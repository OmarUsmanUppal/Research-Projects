import numpy as np
import cv2

# Step 1 Reading from the front camera.
cap = cv2.VideoCapture(0)

# while(True):
#     #Capture Frame by Frame
#     ret, frame=cap.read()
#
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Display the resulting frame
#     cv2.imshow("frame", gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()



# cap.isOpened()
# cap.open()
# cap.get(propId) method where propId is a number from 0 to 18. Each number denotes a property of the video
    # Some of these values can be modified using cap.set(propId, value).

"""
For example, I can check the frame width and height by cap.get(3) and cap.get(4). It gives me 640x480 by default.
But I want to modify it to 320x240. Just use ret = cap.set(3,320) and ret = cap.set(4,240)
"""

# Step 2 Reading the saved video file.
# cap = cv2.VideoCapture('vtest.avi')
# while(cap.isOpened()):
#     ret, frame = cap.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Display the resulting frame
#     cv2.imshow("frame", gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# Step 3 Save video file.
#Video Codecs by FOURCC http://www.fourcc.org/codecs.php

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()