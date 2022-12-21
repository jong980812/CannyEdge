import numpy as np


def encoding_gradient_angle(gradient_angle) :
    ''' input : angle matrix of img gradient
        return : matrix filled with encoded values in [0, 1, 2, 3] according to the angle of gradient '''
    encoding_angle = np.zeros((gradient_angle.shape[0], gradient_angle.shape[1]), dtype=int)
        
    #* divide angle into 4 types
    #* divide 360° into 16 and encoding like this 
    #@ 333332222222211111
    #@ 333333222222111111 
    #@ 033333322221111110
    #@ 000033332211110000
    #@ 000000332211000000
    #@ 000011112233330000
    #@ 011111112233333330
    #@ 111111122223333333
    #@ 111112222222233333 
    #* if encoded angle value = 1, gradient direction is left diagonal
    #* then, look at pixels on the right diagonal and if center pixel is not the maximum, make pixel value 0

    #* supress pixels on the perpendicular side to the gradient direction
    #* to make edges sharp
    
    for i in range(gradient_angle.shape[0]):
        for j in range(gradient_angle.shape[1]):
            if 0 <= gradient_angle[i, j] <= 22.5 or 157.5 <= gradient_angle[i, j] <= 202.5 or 337.5 <= gradient_angle[i, j] <= 360:
                encoding_angle[i, j] = 0
            elif 22.5 <= gradient_angle[i, j] <= 67.5 or 202.5 <= gradient_angle[i, j] <= 247.5:
                encoding_angle[i, j] = 1
            elif 67.5 <= gradient_angle[i, j] <= 112.5 or 247.5 <= gradient_angle[i, j] <= 292.5:
                encoding_angle[i, j] = 2
            elif 112.5 <= gradient_angle[i, j] <= 157.5 or 292.5 <= gradient_angle[i, j] <= 337.5:
                encoding_angle[i, j] = 3
    
    return encoding_angle


#* find perpendicular pixels to gradient directions          #* gradient -> pixels
directions_and_pixels_dict = { 0 : [[0, -1], [0, +1]],       #* horizonal -> vertical
                               1 : [[-1, +1], [+1, -1]],     #* left diagonal -> right diagonal 
                               2 : [[-1, 0], [+1, 0]],       #* vertical -> horizonal
                               3 : [[-1, -1], [+1, +1]] }    #* right diagonal -> left diagonal


def check_am_i_maximum(left_pixel_coords, right_pixel_coords, magnitude) :
    ''' input : left and right pixels's coords list [x, y], img gradient magnitude
        return : assert center pixel is maximum '''
    left_x, left_y = left_pixel_coords
    right_x, right_y = right_pixel_coords
    center_x, center_y = (left_x + right_x) // 2, (left_y + right_y) // 2 
    
    if magnitude[left_x, left_y] <= magnitude[center_x, center_y] and magnitude[right_x, right_y] <= magnitude[center_x, center_y] :    #* if center pixel is maximum
        return True
    else :
        return False
        

def nonmax_suppression(gradient_magnitude, gradient_angle) :
    ''' input : magnitude and angle of img gradient
        return : nonmax suppressed img '''

    suppressed_img = np.zeros((gradient_magnitude.shape[0], gradient_magnitude.shape[1]))    #* initialize

    encoded_angle = encoding_gradient_angle(gradient_angle)    #* encoding gradient's angle into 4 types
    
    for i in range(gradient_magnitude.shape[0] - 1) :
        for j in range(gradient_magnitude.shape[1] - 1) :
            gradient_direction = encoded_angle[i ,j]
            
            left_pixel_coords = [i + directions_and_pixels_dict[gradient_direction][0][0], j + directions_and_pixels_dict[gradient_direction][0][1]]    #* mapping gradient's direction into targeted pixels
            right_pixel_coords = [i + directions_and_pixels_dict[gradient_direction][1][0], j + directions_and_pixels_dict[gradient_direction][1][1]]    
            
            if check_am_i_maximum(left_pixel_coords, right_pixel_coords, gradient_magnitude) :    
                suppressed_img[i, j] = gradient_magnitude[i, j]   
            else :
                suppressed_img[i, j] = 0    #* if center pixel is not the maximum, set to 0
                
    return suppressed_img
            
            
if __name__ == "__main__" :
    import cv2
    import matplotlib.pyplot as plt 
    import argparse
    
    from derivative_of_gaussian import derivative_of_gaussian_filtering as DoG
    from gradient_magnitude_and_angle import get_gradient_magnitude, get_gradient_angle
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str)

    args = parser.parse_args()

    #* in terminal
    #* python nonmax_suppression.py --file_path /data/img.png
    
    img = cv2.imread(args.file_path)
    img_name = args.file_path.split("/")[-1].split(".")[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize = (8,8))
    plt.imsave("img/{}_gray.png".format(img_name), gray, cmap='gray')

    kernel_size, gaussian_std = 7, 1
    gray_DoG_x, gray_DoG_y = DoG(gray, kernel_size, gaussian_std)
    gray_DoG_magnitude = get_gradient_magnitude(gray_DoG_x, gray_DoG_y)
    gray_DoG_angle = get_gradient_angle(gray_DoG_x, gray_DoG_y)
    
    gray_nms = nonmax_suppression(gray_DoG_magnitude, gray_DoG_angle)
    
    plt.imsave("img/{}_gray_DoG_nms_({},{}).png".format(img_name, kernel_size, gaussian_std), gray_nms, cmap='gray')
    

    
    

      