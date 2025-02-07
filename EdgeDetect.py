import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
from skimage import measure

#############################
#       Load The Image      #
#############################

image_pth = "test1.png"
image = plt.imread(image_pth)

#############################################################################
#       Convert image to grayscale and then apply blur using formulae       #  
#############################################################################
     
def convert_to_gray(image):
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2] # Weghted Avg
    return (image[:,:,0]+image[:,:,1]+image[:,:,2])/3 # Normal Avg
gray = convert_to_gray(image)


def convert_to_blur(image , sigma=1):
    return gaussian_filter(gray, sigma)
blur = convert_to_blur(image, 0.1) # the result would also be grayscale

#############################################################
#       Apply Sobel kernel to blurred version of image      #
#############################################################

"""sobel kernel"""
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  
appl_x = cv2.filter2D(src=blur , ddepth=-1 , kernel=sobel_x) # returns numpy array
appl_y = cv2.filter2D(src=blur , ddepth=-1 , kernel=sobel_y) # returns numpy array
sobel_complete = np.sqrt(appl_x**2 + appl_y**2)

sobel_complete = cv2.normalize(sobel_complete, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


######################################################
#       Apply threshold to make a binary photo       #
######################################################
_, thresh = cv2.threshold(sobel_complete, 15, 255, cv2.THRESH_BINARY) 


#############################################
#       Finding contours in the photo       #
#############################################
contours = measure.find_contours(thresh, fully_connected="high") 

# normalizing images : later when drawing contour with open cv we need uint8
#1
if image.dtype != np.uint8:
    image_uint8 = (image * 255).astype(np.uint8) # first mthod of convertion
else:
    image_uint8 = image.copy()
#2
if thresh.dtype != np.uint8:
    thresh_uint8 = (thresh.astype(np.uint8) * 255) # second method
else:
    thresh_uint8 = thresh.copy()

# Creating blank canvases with same dimension as gray scaled version image and with 3 channels

template = (gray.shape[0], gray.shape[1], 3)
all_contours_canvas = np.zeros(template, dtype=np.uint8)
external_contours_canvas = np.zeros(template, dtype=np.uint8)
largest_contour_display = np.zeros(template, dtype=np.uint8)


################################################################################
# We'll store converted contours for cv2 drawing and also find the largest one.#
################################################################################

contours_approximated = []

for contour in contours:
    approx = measure.approximate_polygon(contour, tolerance=2) #tolerance can be adaptive
    approx_cv = approx[:, [1, 0]].reshape(-1, 1, 2).astype(np.int32)
    contours_approximated.append(approx_cv)


largest_contour_cv = max(contours_approximated, key=cv2.contourArea)
ext_contours_appr, _ = cv2.findContours(thresh_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_with_largest = image_uint8.copy()

cv2.drawContours(all_contours_canvas, contours_approximated, -1, (255, 255, 255), 2)
cv2.drawContours(external_contours_canvas, ext_contours_appr, -1, (255, 255, 255), 2)
cv2.drawContours(largest_contour_display, [largest_contour_cv], -1, (0, 0, 255), 10)  
cv2.drawContours(image_with_largest, [largest_contour_cv], -1, (255 , 0, 0), 10)

#####################################
#          Convert To RGB           #
#####################################

def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image_with_largest = bgr_to_rgb(image_with_largest)
largest_contour_display = bgr_to_rgb(largest_contour_display)
image_with_largest = bgr_to_rgb(image_with_largest)

#####################################
#       Display the results         #
#####################################
plt.figure(figsize=(12, 8))

plt.subplot(3, 5, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 5, 2)
plt.imshow(gray,cmap="gray")
plt.title('Gray')
plt.axis('off')

plt.subplot(3, 5, 3)
plt.imshow(blur, cmap='gray')
plt.title('Blurred')
plt.axis('off')

plt.subplot(3, 5, 4)
plt.imshow(appl_x, cmap='gray')
plt.title('sobel x applied')
plt.axis('off')

plt.subplot(3, 5, 5)
plt.imshow(appl_y, cmap='gray')
plt.title('sobel y applied')
plt.axis('off')

plt.subplot(3, 5, 6)
plt.imshow(sobel_complete, cmap='gray')
plt.title('combining sobel x and sobel y')
plt.axis('off')

plt.subplot(3, 5, 7)
plt.imshow(thresh, cmap='gray')
plt.title("threshold on final sobel")
plt.axis('off')

plt.subplot(3, 5, 8)
plt.imshow(all_contours_canvas)
plt.title("All Contours")
plt.axis('off')

plt.subplot(3, 5, 9)
plt.imshow(external_contours_canvas)
plt.title("External Contours Approx")
plt.axis('off')

plt.subplot(3, 5, 10)
plt.imshow(largest_contour_display)
plt.title("Largest Contour")
plt.axis('off')

plt.subplot(3, 5, 11)
plt.imshow(image_with_largest)
plt.title("Largest Outline Overlay")
plt.axis('off')

plt.tight_layout()
plt.show()
