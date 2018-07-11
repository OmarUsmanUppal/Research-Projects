import numpy as np
import cv2

x = np.uint8([250])
y = np.uint8([10])

print(x,y)

print (cv2.add(x,y)) #250+10 = 255
print(x+y) #250+10 = 260/256 = 4

# OpenCV addition is a saturated operation while Numpy addition is a modulo operation.
#
# img1 = cv2.imread('ml.png')
# img2 = cv2.imread('opencv_logo.jpg')
#
# dst = cv2.addWeighted(img1,0.7,img2,0.3,0)
#
# cv2.imshow('dst',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""Bitwise Operations
This includes bitwise AND, OR, NOT and XOR operations.
They will be highly useful while extracting any part of the image (as we will see in coming chapters),
        defining and working with non-rectangular ROI etc.
Below we will see an example on how to change a particular region of an image.
"""

# Load two images
img1 = cv2.imread('..\\messi5.jpg')
img2 = cv2.imread('..\\opencv_logo.jpg')

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst

cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()