#!/usr/bin/evn python

# Author: Chinmaya Khamesra 

# Dependencies 
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Harris corner function
image_path = "/home/chinmay/Downloads/11.jpeg"
kernel_size = 5
k = 0.04
threshold = 0.30

# Reading the image from the device 
image = cv2.imread(image_path)
# Converting the rgb image into black/white 
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Gaussian kernel 
gaussian = cv2.GaussianBlur(gray,(3,3),0)

# Dimension of the image
h, w = image.shape[0] , image.shape[1]
# Null matrix 
matrix = np.zeros((h,w))

# Derivative calculations (1st and 2nd derivative)
x_1 = cv2.Sobel(gaussian, cv2.CV_64F, 1, 0, ksize=3)
y_1 = cv2.Sobel(gaussian, cv2.CV_64F, 0, 1, ksize=3)
x_2, y_2=np.square(x_1), np.square(y_1)

dxy = x_1 * y_1

offset = int( kernel_size / 2 )

# Corner determinentection 
print("Starting finding the corners")

for j in range(offset, h-offset):
    for i in range(offset, w-offset):

        Sobelx = np.sum(x_2[j-offset:j+1+offset, i-offset:i+1+offset])
        Sobely = np.sum(y_2[j-offset:j+1+offset, i-offset:i+1+offset])
        Sobelxy = np.sum(dxy[j-offset:j+1+offset, i-offset:i+1+offset])

        # Evaluating the matrix 
        harris = np.array([[Sobelx,Sobelxy],[Sobelxy,Sobely]])
        determinent=np.linalg.det(harris)
        trace=np.matrix.trace(harris)

        # Applying in the formula 
        R=determinent-k*(trace**2)
        matrix[j-offset, i-offset]=R

# Thresholding 
cv2.normalize(matrix, matrix, 0, 1, cv2.NORM_MINMAX)
for j in range(offset, h-offset):
    for i in range(offset, w-offset):
        value=matrix[j, i]
        if value>threshold:
            cv2.circle(image,(i,j),3,(255,0,0))
        
# Plotting the figure 
plt.figure("Harris Corner determinentector")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Harris Corner determinentector")
plt.show()

