import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
#argument를 출력해주는 함수,
def arg_print(args):
    print("-------------------------------Argument------------------------------------")
    for k, v in vars(args).items():
        print(f'{k}={v}')
    print("---------------------------------------------------------------------------")


#이미지를 패딩해주는 함수.
def Get_paddig_image(img:np.ndarray, filter_scope:int, filter_size:int):
    '''
    Padding image
    #@ img: Image
    #@ filter_scope: number of add term each side
    패딩하는 이유는, 이미지를 필터링 할 때 외곽에 있는 원소들은 존재하지 않는 값에 연산을 해야하므로,
    0으로 패딩하여 연산 한다. 
    패딩 방법은 0으로 패딩 사이즈 만큼 초기화 해놓고, 원본을 다시 대입하는 방식.
    '''
    empty_space=np.zeros((filter_size-1+img.shape[0], filter_size-1+img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            empty_space[filter_scope + i][ filter_scope + j] = img[i][j]
    return empty_space

def MakeGrid(size : int)->np.ndarray:
    '''
    #@ size: number of grid width,height -> filter_scope
    #@ np.meshgrid make grid each axis( x, y )
    '''
    x,y = np.meshgrid(range(-int(size), int(size) + 1), range(-int(size), int(size) + 1))
    return x,y

def Show_smoothing_img(original, transform:np.ndarray)->None:
    '''
    Show plt
    원본이미지와 smooth이미지를 동시에 출력해주는 함수.
    '''
    fig = plt.figure(figsize=(15,15))
    img1 = fig.add_subplot(1, 2, 1)
    img1.imshow(original, cmap='gray')

    img2= fig.add_subplot(1,2,2)
    img2.imshow(transform,cmap='gray')
    plt.show()
    
def Show_img(original):
    plt.figure(figsize=(10,10))
    plt.imshow(original, cmap='gray')
    plt.show()
    #이미지 보여주기
    
def Save_img(original,path,tag):
    plt.imsave(path+f'_{tag}.jpg',original,cmap='gray')
    # 이미지 저장. 
def print_bar():
    print("--------------------------------------------------------------")