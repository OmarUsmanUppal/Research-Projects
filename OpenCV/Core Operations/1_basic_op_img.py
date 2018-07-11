import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('..\\messi5.jpg')

px = img[100,100]
print (px)

# accessing only blue pixel
blue = img[100,100,0]
print (blue)

img[100,100] = [255,255,255]
print (img[100,100])


# Better pixel accessing and editing method :
"""
For individual pixel access, Numpy array methods, array.item() and array.itemset() is considered to be better.
But it always returns a scalar.
So if you want to access all B,G,R values, you need to call array.item() separately for all.
"""
print(img.item(10,10,2))
img.itemset((10,10,2),100)

print(img.item(10,10,2))

"""
Image properties include number of rows, columns and channels, type of image data, number of pixels etc.
Shape of image is accessed by img.shape.
It returns a tuple of number of rows, columns and channels (if image is color):

If image is grayscale, tuple returned contains only number of rows and columns.
So it is a good method to check if loaded image is grayscale or color image.
"""
print ("image shape",img.shape)

print (img.size)
print (img.dtype)

ball = img[280:340, 330:390]
img[273:333, 100:160] = ball

cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Splitting and Merging Image Channels
b,g,r = cv2.split(img)
img = cv2.merge((b,g,r))

"""OR
b = img[:,:,0]

NOTE: cv2.split() is a costly operation (in terms of time), so only use it if necessary.
 Numpy indexing is much more efficient and should be used if possible.
"""

"""Making Borders for Images (Padding)
If you want to create a border around the image, something like a photo frame,
        you can use cv2.copyMakeBorder() function.
But it has more applications for convolution operation, zero padding etc.
This function takes following arguments:
"""


BLUE = [255,0,0]

img1 = cv2.imread('..\\opencv_logo.jpg')

replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()