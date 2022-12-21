import cv2
import matplotlib.pyplot as plt 
import argparse

from gaussian import gaussian_smoothing
from derivative_of_gaussian import derivative_of_gaussian_filtering 
from gradient_magnitude_and_angle import get_gradient_magnitude, get_gradient_angle
from nonmax_suppression import nonmax_suppression
from double_thresholding import double_thresholding
from hysteresis import hysteresis

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str)
parser.add_argument("-k", "--gaussian_kernel_size", type=int)
parser.add_argument("-s", "--gaussian_std", type=int)
parser.add_argument("-l", "--low_threshold", type=float)
parser.add_argument("-hi", "--high_threshold", type=float)
parser.add_argument("-r", "--hysteresis_range", type=int)   

args = parser.parse_args()

#* in terminal
#* python main.py --file_path /data/img.png -k 7 -s 1 -l 10 -hi 30 -r 1

#* use opencv to load image file
img = cv2.imread(args.file_path)
img_name = args.file_path.split("/")[-1].split(".")[0]

#* convert image to grayscale by using opencv
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#* use matplotlib to save image file
plt.figure(figsize = (8,8))
plt.imsave("img/{}_gray.png".format(img_name), gray, cmap='gray')

#* hyperparameter setting
kernel_size, gaussian_std = args.gaussian_kernel_size, args.gaussian_std   #* for gaussian smoothing and DoG filtering
low_threshold, high_threshold = args.low_threshold, args.high_threshold    #* for double thresholding
hysteresis_range = args.hysteresis_range                                   #* for hysteresis
     
#* for saved img name
low_name = args.low_threshold if int(args.low_threshold) == 0 else int(args.low_threshold) 
high_name = args.high_threshold if int(args.high_threshold) == 0 else int(args.high_threshold) 

#*##############################################################################################################################
#* Canny Edge Detector

#* 1. gaussian smoothing
#*    to smooth noise pixels
gray_gaussian = gaussian_smoothing(gray, kernel_size, gaussian_std)
plt.imsave("img/{}_gray_guassian_({},{}).png".format(img_name, kernel_size, gaussian_std), gray_gaussian, cmap='gray')

#* 2. derivative of gaussian filtering to get gradients of image pixels
#*    to detect the edges
gray_DoG_x, gray_DoG_y = derivative_of_gaussian_filtering(gray, kernel_size, gaussian_std)
plt.imsave("img/{}_gray_DoG_x_({},{}).png".format(img_name, kernel_size, gaussian_std), gray_DoG_x, cmap='gray')
plt.imsave("img/{}_gray_DoG_y_({},{}).png".format(img_name, kernel_size, gaussian_std), gray_DoG_y, cmap='gray')

#* 3. get magnitude and angle of gradients of pixel values
#*    for thresholding
gray_DoG_magnitude = get_gradient_magnitude(gray_DoG_x, gray_DoG_y)
gray_DoG_angle = get_gradient_angle(gray_DoG_x, gray_DoG_y)
plt.imsave("img/{}_gray_DoG_magnitude_({},{}).png".format(img_name, kernel_size, gaussian_std), gray_DoG_magnitude, cmap='gray')
plt.imsave("img/{}_gray_DoG_angle_({},{}).png".format(img_name, kernel_size, gaussian_std), gray_DoG_angle, cmap='gray')

#* 4. nonmax-suppression 
#*    to make the edges sharp
gray_nms = nonmax_suppression(gray_DoG_magnitude, gray_DoG_angle)
plt.imsave("img/{}_gray_DoG_nms_({},{}).png".format(img_name, kernel_size, gaussian_std), gray_nms, cmap='gray')

#* 5. double thresholding
#*    to classify pixels into {edge, not edge, maybe edge}
gray_double_thresholding = double_thresholding(gray_nms, low_threshold, high_threshold)
plt.imsave("img/{}_gray_thresholding_({},{})_({},{}).png".format(img_name, kernel_size, gaussian_std, low_name, high_name), gray_double_thresholding, cmap='gray')
#! print(gray_double_thresholding[120])

#* 6. hyteresis
#*    to judge the edgeness of ambiguous pixel with adjacent pixels' edgeness
gray_hysteresis = hysteresis(gray_double_thresholding, hysteresis_range)
plt.imsave("img/{}_gray_hysteresis_({},{})_({},{})_{}.png".format(img_name, kernel_size, gaussian_std, low_name, high_name, hysteresis_range), gray_hysteresis, cmap='gray')
#! print(gray_hysteresis[120])
