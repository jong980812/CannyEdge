import numpy as np
import cv2
import matplotlib.pyplot as plt

def two_dim_gaussian_function(x, y, std) -> float :
    return np.exp(-1 * (x**2 + y**2) / (2 * std**2)) / (2 * np.pi * std**2)


def gaussian_filter(kernel_size, std) -> np.array(int) :  
    g_filter = np.zeros((kernel_size, kernel_size))
    half_size = kernel_size // 2
    for i, x in enumerate(range(-1 * half_size, half_size + 1)) :
        for j, y in enumerate(range(-1 * half_size, half_size + 1)) : 
            g_filter[i, j] = two_dim_gaussian_function(x, y, std)  #* g_filter[0, 0] = gaussian(-2, -2, std) if k = 3, 가우시안 필터 좌측 상단 첫번쨰 원소로는 원점에서 x축, y축 -2씩 떨어져있는 지점에서의 가우시안 값이 들어 감
    
    return g_filter

def padding(img, padding_size, value) -> np.array :
    img_h = img.shape[0]  #* img.shape = (H, W, C)
    img_w = img.shape[1]
    padded_img = np.full((img_h + 2 * padding_size, img_w + 2 * padding_size), value)
    padded_img[padding_size : -1 * padding_size, padding_size : -1 * padding_size] = img
    return padded_img
     

def gaussian_smoothing(img, kernel_size, guassian_std) -> np.array :
    #* smoothing = linear filtering = convolution
    g_filter = gaussian_filter(kernel_size, guassian_std)
    
    #* to remain same img size, do padding
    #* n = input img size, f = filter size, p = one-side padding size
    #* n + 2 * p - f + 1 = n  ->  p = (f - 1) / 2
    padding_size = (kernel_size - 1) // 2  #* / -> float, // -> int
    padded_img = padding(img, padding_size, 0)
    
    img_h, img_w = img.shape[0], img.shape[1]
    smoothed_img = np.zeros((img_h, img_w))
    
    for i in range(img_h) :
        for j in range(img_w) :
            smoothed_img[i, j] = np.sum(padded_img[i : i + kernel_size, j : j + kernel_size] * g_filter)   #* element-wise multiplication = * in numpy
            
    return smoothed_img
    
image = cv2.imread('/data/ahngeo11/edge/img/chess_board-svg.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure(figsize = (8,8))
plt.imsave("img/chess_board_gray.png", gray, cmap='gray')

kernel_size, gaussian_std = 3, 10
gray_gaussian = gaussian_smoothing(gray, kernel_size, gaussian_std)
plt.imsave("img/chess_board_gray_guassian_({},{}).png".format(kernel_size, gaussian_std), gray_gaussian, cmap='gray')
    