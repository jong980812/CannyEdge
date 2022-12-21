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
    
    if magnitude[left_x, left_y] <= magnitude[center_x, center_y] or magnitude[right_x, right_y] <= magnitude[center_x, center_y] :    #* if center pixel is maximum
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

#!#########################################################################################################################################

def encoding_gradient_angle_8types(gradient_angle) :
    ''' input : angle matrix of img gradient
        return : matrix filled with encoded values in [0, 1, 2, 3, 4, 5, 6, 7] according to the angle of gradient '''
    encoding_angle = np.zeros((gradient_angle.shape[0], gradient_angle.shape[1]), dtype=int)
        
    #* divide angle into 8 types
    #* divide 360° into 32 and encoding like this 
    
    for i in range(gradient_angle.shape[0]):
        for j in range(gradient_angle.shape[1]):
            if 0 <= gradient_angle[i, j] <= 11.25 or 348.75 <= gradient_angle[i, j] <= 360 or 168.75 <= gradient_angle[i, j] <= 191.25 :
                encoding_angle[i, j] = 0
            elif 11.25 <= gradient_angle[i, j] <= 33.75 or 191.25 <= gradient_angle[i, j] <= 213.75:
                encoding_angle[i, j] = 1
            elif 33.75 <= gradient_angle[i, j] <= 56.25 or 213.75 <= gradient_angle[i, j] <= 236.25:
                encoding_angle[i, j] = 2
            elif 56.25 <= gradient_angle[i, j] <= 78.75 or 236.25 <= gradient_angle[i, j] <= 258.75:
                encoding_angle[i, j] = 3
            elif 78.25 <= gradient_angle[i, j] <= 101.25 or 258.75 <= gradient_angle[i, j] <= 281.25:
                encoding_angle[i, j] = 4
            elif 101.25 <= gradient_angle[i, j] <= 123.75 or 281.25 <= gradient_angle[i, j] <= 303.75:
                encoding_angle[i, j] = 5
            elif 123.75 <= gradient_angle[i, j] <= 146.25 or 303.75 <= gradient_angle[i, j] <= 326.25:
                encoding_angle[i, j] = 6
            elif 146.25 <= gradient_angle[i, j] <= 168.75 or 326.25 <= gradient_angle[i, j] <= 348.75:
                encoding_angle[i, j] = 7
    
    return encoding_angle


directions_and_pixels_dict_with_8types = { 0 : [[0, -1], [0, +1]],       #* horizonal -> vertical
                                           1 : [0, 2],                   #* use interpolated pixels
                                           2 : [[-1, +1], [+1, -1]],     #* left diagonal -> right diagonal
                                           3 : [2, 4],                   #* use interpolated pixels
                                           4 : [[-1, 0], [+1, 0]],       #* vertical -> horizonal
                                           5 : [4, 6],                   #* use interpolated pixels
                                           6 : [[-1, -1], [+1, +1]],     #* right diagonal -> left diagonal  
                                           7 : [6, 0]}                   #* use interpolated pixels


def check_am_i_maximum_interpolation_pixels_ver(first_coords_for_left_pixel, second_coords_for_left_pixel, first_coords_for_right_pixel, second_coords_for_right_pixel, magnitude) :
    ''' input : left and right pixels's coords list [x, y], img gradient magnitude
        return : assert center pixel is maximum '''
    x1, y1 = first_coords_for_left_pixel
    x2, y2 = second_coords_for_left_pixel
    x3, y3 = first_coords_for_right_pixel
    x4, y4 = second_coords_for_right_pixel
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2 
    
    #* get magnitude by interpolation
    left_pixel_magnitude = (magnitude[x1, y1] + magnitude[x2, y2]) / 2
    right_pixel_magnitude = (magnitude[x3, y3] + magnitude[x4, y4]) / 2
    
    if left_pixel_magnitude <= magnitude[center_x, center_y] or right_pixel_magnitude <= magnitude[center_x, center_y] :    #* if center pixel is maximum
        return True
    else :
        return False


def nonmax_suppression_with_8types_encoding(gradient_magnitude, gradient_angle) :
    ''' input : magnitude and angle of img gradient
        return : nonmax suppressed img '''

    suppressed_img = np.zeros((gradient_magnitude.shape[0], gradient_magnitude.shape[1]))    #* initialize

    encoded_angle = encoding_gradient_angle(gradient_angle)    #* encoding gradient's angle into 4 types
    
    for i in range(gradient_magnitude.shape[0] - 1) :
        for j in range(gradient_magnitude.shape[1] - 1) :
            gradient_direction = encoded_angle[i ,j]
            
            mapped_pixels = directions_and_pixels_dict_with_8types[gradient_direction]
            
            if type(mapped_pixels[0]) == list :
                left_pixel_coords = [i + mapped_pixels[0][0], j + mapped_pixels[0][1]]    #* mapping gradient's direction into targeted pixels
                right_pixel_coords = [i + mapped_pixels[1][0], j + mapped_pixels[1][1]]    
            
                #* nonmax suppression
                if check_am_i_maximum(left_pixel_coords, right_pixel_coords, gradient_magnitude) :    
                    suppressed_img[i, j] = gradient_magnitude[i, j]   
                else :
                    suppressed_img[i, j] = 0    #* if center pixel is not the maximum, set to 0
            
            elif type(mapped_pixels[0]) == int :
                #* do interpolation to get perpendicular pixels to gradient direction
                
                #* get directions to be interpolated
                first_interpolation_direction, second_interpolation_direction = mapped_pixels 
                
                #* get pixels coords 
                first_coords_for_left_pixel = [i + directions_and_pixels_dict_with_8types[first_interpolation_direction][0][0], j + directions_and_pixels_dict_with_8types[first_interpolation_direction][0][1]]
                second_coords_for_left_pixel = [i + directions_and_pixels_dict_with_8types[first_interpolation_direction][1][0], j + directions_and_pixels_dict_with_8types[first_interpolation_direction][1][1]]
                first_coords_for_right_pixel = [i + directions_and_pixels_dict_with_8types[second_interpolation_direction][0][0], j + directions_and_pixels_dict_with_8types[second_interpolation_direction][0][1]]
                second_coords_for_right_pixel = [i + directions_and_pixels_dict_with_8types[second_interpolation_direction][1][0], j + directions_and_pixels_dict_with_8types[second_interpolation_direction][1][1]]
                
                #* nonmax suppression
                if check_am_i_maximum_interpolation_pixels_ver(first_coords_for_left_pixel, second_coords_for_left_pixel, first_coords_for_right_pixel, second_coords_for_right_pixel, gradient_magnitude) :    
                    suppressed_img[i, j] = gradient_magnitude[i, j]   
                else :
                    suppressed_img[i, j] = 0
            
                
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

    kernel_size, gaussian_std = 3, 10
    gray_DoG_x, gray_DoG_y = DoG(gray, kernel_size, gaussian_std)
    gray_DoG_magnitude = get_gradient_magnitude(gray_DoG_x, gray_DoG_y)
    gray_DoG_angle = get_gradient_angle(gray_DoG_x, gray_DoG_y)
    
    gray_nms = nonmax_suppression(gray_DoG_magnitude, gray_DoG_angle)
    
    plt.imsave("img/{}_gray_DoG_nms_({},{}).png".format(img_name, kernel_size, gaussian_std), gray_nms, cmap='gray')
    
    gray_nms = nonmax_suppression_with_8types_encoding(gray_DoG_magnitude, gray_DoG_angle)
    plt.imsave("img/{}_gray_DoG_nms_8types_({},{}).png".format(img_name, kernel_size, gaussian_std), gray_nms, cmap='gray')
    plt.imsave("img/{}_gray_DoG_magnitude_({},{}).png".format(img_name, kernel_size, gaussian_std), gray_DoG_magnitude, cmap='gray')
    
    