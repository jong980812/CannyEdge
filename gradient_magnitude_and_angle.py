import numpy as np


def gradient_magnitude(fx, fy) -> np.ndarray :
    return np.sqrt(np.power(fx, 2) + np.power(fy, 2))
    
    
def gradient_angle(fx, fy) -> np.ndarray :
    return np.arctan2(fy, fx)


if __name__ == "__main__" :
    import cv2
    import matplotlib.pyplot as plt 
    import argparse
    
    from derivative_of_gaussian import derivative_of_gaussian_filtering as DoG
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str)

    args = parser.parse_args()

    #* in terminal
    #* python gradient_magnitude_and_angle.py --file_path /data/img.png
    
    img = cv2.imread(args.file_path)
    img_name = args.file_path.split("/")[-1].split(".")[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize = (8,8))
    plt.imsave("img/{}_gray.png".format(img_name), gray, cmap='gray')

    kernel_size, gaussian_std = 7, 1
    gray_DoG_x, gray_DoG_y = DoG(gray, kernel_size, gaussian_std)
    gray_DoG_magnitude = gradient_magnitude(gray_DoG_x, gray_DoG_y)
    gray_DoG_angle = gradient_angle(gray_DoG_x, gray_DoG_y)
    plt.imsave("img/{}_gray_DoG_magnitude_({},{}).png".format(img_name, kernel_size, gaussian_std), gray_DoG_magnitude, cmap='gray')
    plt.imsave("img/{}_gray_DoG_angle_({},{}).png".format(img_name, kernel_size, gaussian_std), gray_DoG_angle, cmap='gray')
    
    #!###########################################
    #! like CannyEdgeDetector.ipynb, smoothing -> DoG
    from gaussian import gaussian_smoothing
    gray_smoothing = gaussian_smoothing(gray, kernel_size, gaussian_std)
    gray_DoG_x, gray_DoG_y = DoG(gray_smoothing, kernel_size, gaussian_std)
    gray_DoG_magnitude = gradient_magnitude(gray_DoG_x, gray_DoG_y)
    gray_DoG_angle = gradient_angle(gray_DoG_x, gray_DoG_y)
    plt.imsave("img/{}_gray_smoothing_DoG_magnitude_({},{}).png".format(img_name, kernel_size, gaussian_std), gray_DoG_magnitude, cmap='gray')
    plt.imsave("img/{}_gray_smoothing_DoG_angle_({},{}).png".format(img_name, kernel_size, gaussian_std), gray_DoG_angle, cmap='gray')
    #!###########################################
    