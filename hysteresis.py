import numpy as np

def generate_adjacent_pixels_coords(hysteresis_range) :
    ''' input : range to check there is edge
        return : list of tuple of coords of adjacent pixels in range '''
    adjacent_pixels_coords = []
    
    #* if range = 1, len(adjacent_pixels_coords) = 8
    for i in range(-1 * hysteresis_range, hysteresis_range + 1) :
        for j in range(-1 * hysteresis_range, hysteresis_range + 1) :
            adjacent_pixels_coords.append((i, j)) 
    adjacent_pixels_coords.remove((0, 0))    #* remove myself
    
    return adjacent_pixels_coords
    
def hysteresis(thresholded_img, hysteresis_range) :
    ''' input : double thresholded img, range to check there is edge
        return : hysteresis double thresholded img '''
    hysteresis_img = np.zeros((thresholded_img.shape[0], thresholded_img.shape[1]))
    
    adjacent_pixels_coords = generate_adjacent_pixels_coords(hysteresis_range)

    for i in range(thresholded_img.shape[0] - hysteresis_range) :
        for j in range(thresholded_img.shape[1] - hysteresis_range) :
            if thresholded_img[i, j] == 255.0 :    #* if the pixel is edge
                hysteresis_img[i, j] = 255.0      #* keep
            elif thresholded_img[i, j] == 128.0 :    #* if the pixel is not sure
                for x, y in adjacent_pixels_coords :    #* check adjacent pixels
                    if thresholded_img[i + x, j + y] == 255.0 :    #* if the edge is near
                        hysteresis_img[i, j] = 255.0     #* this pixel be also edge
                        break 
            #* if the pixel is surely not edge or edges are not near, the pixel set to 0
    
    return hysteresis_img        


if __name__ == "__main__" :
    import cv2
    import matplotlib.pyplot as plt 
    import argparse
    
    from derivative_of_gaussian import derivative_of_gaussian_filtering as DoG
    from gradient_magnitude_and_angle import get_gradient_magnitude, get_gradient_angle
    from nonmax_suppression import nonmax_suppression
    from double_thresholding import double_thresholding
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str)
    parser.add_argument("-k", "--gaussian_kernel_size", type=int)
    parser.add_argument("-s", "--gaussian_std", type=int)
    parser.add_argument("-l", "--low_threshold", type=float)
    parser.add_argument("-hi", "--high_threshold", type=float)
    parser.add_argument("-r", "--hysteresis_range", type=int)   
    
    args = parser.parse_args()

    #* in terminal
    #* python hysteresis.py --file_path /data/img.png -k 7 -s 1 -l 10 -hi 30 -r 1
    
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
    #! print(gray_double_thresholding[120])
    gray_hysteresis = hysteresis(gray_double_thresholding, args.hysteresis_range)
    #! print(gray_hysteresis[120])
    
    #* for saved img name
    low_name = args.low_threshold if int(args.low_threshold) == 0 else int(args.low_threshold) 
    high_name = args.high_threshold if int(args.high_threshold) == 0 else int(args.high_threshold) 
        
    plt.imsave("img/{}_gray_thresholding_({},{})_({},{}).png".format(img_name, kernel_size, gaussian_std, low_name, high_name), gray_double_thresholding, cmap='gray')
    plt.imsave("img/{}_gray_hysteresis_({},{})_({},{})_{}.png".format(img_name, kernel_size, gaussian_std, low_name, high_name, args.hysteresis_range), gray_hysteresis, cmap='gray')
    