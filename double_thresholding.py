import numpy as np

def double_thresholding(suppressed_img, low_threshold, high_threshold) :
    ''' input : gradient magnitude of differentiated and suppressed img, low and high threshold for magnitude 
        return : double thresholded img ''' 
    thresholded_img = np.zeros((suppressed_img.shape[0], suppressed_img.shape[1]))
    
    for i in range(suppressed_img.shape[0]) :
        for j in range(suppressed_img.shape[1]) :
            if suppressed_img[i, j] < low_threshold :    #* definitely not edge
                thresholded_img[i, j] = 0.0
            elif low_threshold <= suppressed_img[i, j] < high_threshold :    #* maybe edge
                thresholded_img[i, j] = 128.0
            elif suppressed_img[i, j] >= high_threshold :    #* definitely edge
                thresholded_img[i, j] = 255.0
    
    return thresholded_img 


if __name__ == "__main__" :
    import cv2
    import matplotlib.pyplot as plt 
    import argparse
    
    from derivative_of_gaussian import derivative_of_gaussian_filtering as DoG
    from gradient_magnitude_and_angle import get_gradient_magnitude, get_gradient_angle
    from nonmax_suppression import nonmax_suppression
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str)
    parser.add_argument("-k", "--gaussian_kernel_size", type=int)
    parser.add_argument("-s", "--gaussian_std", type=int)
    parser.add_argument("-l", "--low_threshold", type=float)
    parser.add_argument("-hi", "--high_threshold", type=float)
    
    args = parser.parse_args()

    #* in terminal
    #* python double_thresholding.py --file_path /data/img.png -k 7 -s 1 -l 10 -hi 30
    
    img = cv2.imread(args.file_path)
    img_name = args.file_path.split("/")[-1].split(".")[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize = (8,8))
    plt.imsave("img/{}_gray.png".format(img_name), gray, cmap='gray')

    kernel_size, gaussian_std = args.gaussian_kernel_size, args.gaussian_std
    gray_DoG_x, gray_DoG_y = DoG(gray, kernel_size, gaussian_std)
    gray_DoG_magnitude = get_gradient_magnitude(gray_DoG_x, gray_DoG_y)
    gray_DoG_angle = get_gradient_angle(gray_DoG_x, gray_DoG_y)
    
    gray_nms = nonmax_suppression(gray_DoG_magnitude, gray_DoG_angle)
    gray_double_thresholding = double_thresholding(gray_nms, args.low_threshold, args.high_threshold)
    
    #* for saved img name
    low_name = args.low_threshold if int(args.low_threshold) == 0 else int(args.low_threshold) 
    high_name = args.high_threshold if int(args.high_threshold) == 0 else int(args.high_threshold) 
    
    plt.imsave("img/{}_gray_thresholding_({},{})_({},{}).png".format(img_name, kernel_size, gaussian_std, low_name, high_name), gray_double_thresholding, cmap='gray')
    