import numpy as np
from gaussian import padding


def derivative_of_2d_gaussian_by_x(x, y, std) -> float : 
    return np.exp(-1 * (x**2 + y**2) / (2 * std**2)) / (2 * np.pi * std**4) * (-1 * x)    #* partial first derivative of 2d gaussian by x 

def derivative_of_2d_gaussian_by_y(x, y, std) -> float :
    return np.exp(-1 * (x**2 + y**2) / (2 * std**2)) / (2 * np.pi * std**4) * (-1 * y)    #* partial first derivative of 2d gaussian by y


def generate_DoG_filter(kernel_size, std) -> "(np.ndarray, np.ndarray)" :
    DoG_x_filter = np.zeros((kernel_size, kernel_size))
    DoG_y_filter = np.zeros((kernel_size, kernel_size))
    half_size = kernel_size // 2
    for i, x in enumerate(range(-1 * half_size, half_size + 1)) :
        for j, y in enumerate(range(-1 * half_size, half_size + 1)) : 
            DoG_x_filter[i, j] = derivative_of_2d_gaussian_by_x(x, y, std)
            DoG_y_filter[i, j] = derivative_of_2d_gaussian_by_y(x, y, std)
    
    return DoG_x_filter, DoG_y_filter


#* derivative(smoothing(img)) = derivative of gaussian(img)
def derivative_of_gaussian_filtering(img, kernel_size, gaussian_std) -> "(np.ndarray, np.ndarray)" :
    DoG_x_filter, DoG_y_filter = generate_DoG_filter(kernel_size, gaussian_std)
    
    padding_size = (kernel_size - 1) // 2
    padded_img = padding(img, padding_size, 0)
    
    img_h, img_w = img.shape[0], img.shape[1]
    DoG_x_img = np.zeros((img_h, img_w))
    DoG_y_img = np.zeros((img_h, img_w))
    
    #* filtering (convolution)
    for i in range(img_h) :
        for j in range(img_w) :
            DoG_x_img[i, j] = np.sum(padded_img[i : i + kernel_size, j : j + kernel_size] * DoG_x_filter)
            DoG_y_img[i, j] = np.sum(padded_img[i : i + kernel_size, j : j + kernel_size] * DoG_y_filter)  
            
    return DoG_x_img, DoG_y_img


if __name__ == "__main__" :
    import cv2
    import matplotlib.pyplot as plt 
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--kernel_size", type=int)
    parser.add_argument("--gaussian_std", type=int)

    args = parser.parse_args()

    #* in terminal
    #* python derivative_of_gaussian.py --file_path /data/img.png --kernel_size 3 --gaussian_std 10
    
    img = cv2.imread(args.file_path)
    img_name = args.file_path.split("/")[-1].split(".")[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize = (8,8))
    plt.imsave("img/{}_gray.png".format(img_name), gray, cmap='gray')

    kernel_size, gaussian_std = 3, 10
    gray_DoG_x, gray_DoG_y = derivative_of_gaussian_filtering(gray, kernel_size, gaussian_std)
    plt.imsave("img/{}_gray_DoG_x_({},{}).png".format(img_name, kernel_size, gaussian_std), gray_DoG_x, cmap='gray')
    plt.imsave("img/{}_gray_DoG_y_({},{}).png".format(img_name, kernel_size, gaussian_std), gray_DoG_y, cmap='gray')
    