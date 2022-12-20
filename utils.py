import numpy as np
import matplotlib.pyplot as plt
import cv2
def arg_print(args):
    print("-------------------------------Argument------------------------------------")
    for k, v in vars(args).items():
        print(f'{k}={v}')
    print("---------------------------------------------------------------------------")


def Get_paddig_image(img:np.ndarray, filter_scope:int, filter_size:int):
    '''
    Padding image
    #@ img: Image
    #@ filter_scope: number of add term each side
    
    '''
    empty_space=np.zeros((filter_size-1+img.shape[0], filter_size-1+img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            empty_space[filter_scope + i][ filter_scope + j] = img[i][j]
    return empty_space


def Show_smoothing_img(original, transform:np.ndarray)->None:
    '''
    Show plt
    '''
    fig = plt.figure(figsize=(15,15))
    img1 = fig.add_subplot(1, 2, 1)
    img1.imshow(original, cmap='gray')

    img2= fig.add_subplot(1,2,2)
    img2.imshow(transform,cmap='gray')
    plt.show()
    
