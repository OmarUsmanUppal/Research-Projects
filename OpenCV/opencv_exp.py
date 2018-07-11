import numpy as np
import cv2

# Load an color image in grayscale
# img = cv2.imread('messi5.jpg',0)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imwrite('messigray.png',img)

""" READ IMAGE ---SAVE IMAGE on s and exist without saving on ESC"""
img = cv2.imread('messi5.jpg',0)
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()