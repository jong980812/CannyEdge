import numpy as np
from utils import Get_paddig_image, MakeGrid
PI=np.pi

def Get_Gaussian_2d(x:np.ndarray=None ,y:np.ndarray=None ,sigma=None)->np.ndarray:
    '''
    Gaussian 2d function
    Matrix calculation 2d
    #@ x: grid x
    #@ y: grid y
    가우시안 2d 함수를 얻는 코드이다. 
    교과서와 다르게 정석대로 1/2pi*sigma를 곱했다.
    입력은 x, y grid좌표로 들어와서 각 좌표에 맞는 가우시안값을 출력한다.
    '''
    scale_term = 1 / ( 2* PI * sigma**2) # gaussian coefficient
    exp_term = ((x ** 2) + (y ** 2)) / (2 * (sigma ** 2)) # exp term
    gaussian=scale_term * np.exp(-exp_term)
    return gaussian
    
    
def Gaussian_filter(size:int, sigma:float)->np.ndarray:
    '''
    #@ size: filter_scope
    #@ sigma: Standard deviation
    가우시안 2d 필터를 얻는 코드.
    '''
    grid_x, grid_y= MakeGrid(size)                          #@ Get Grid ( like 2d coordinate )
    gaussian_filter = Get_Gaussian_2d(grid_x,grid_y,sigma)  #@ Get Gaussian 2d depend on grid size, sigma
    
    return gaussian_filter

def Box_filter(size:int):
    '''
    #@ size: filter_size
    box filter를 얻는다. 
    박스 원소 개수만큼 나누어서 합이 1이 되게 한다.
    '''
    box = np.ones((size,size)) # box filter have element '1' 
    box_filter = box/(size**2) # Normalize 
    return box_filter



def Get_filter(method:str, sigma:float, filter_scope:int, filter_size:int)->np.ndarray:
    '''
    정해진 필터를 출력하고, 필터 정보를 피드백 해준다. 
    '''
    if method == "box":
        filter = Box_filter(filter_size)
    elif method == "gaussian":
        filter = Gaussian_filter(filter_scope, sigma)
    print(f"Filter Scope is {filter_scope}\nFilter size is {filter_size}\nSmoothing {method} Filter ") #! print Filter Information
    return filter

def Smoothing(
            img,
            method:str, 
            sigma:float,
            filter_scope:int,
            filter_size:int,
              ):
    '''
    #@ Convolution Weighted sum.
    #@ Vectorized Dot

    Smoothing을 관장하는 함수
    이미지와, smoothing method, sigma를 받아서 smoothing 이미지를 출력한다. 
    '''
    smooth_img=np.zeros_like(img) # 원본이미지와 같은 크기로 초기화.
    filter=Get_filter(method, sigma, filter_scope, filter_size) # 필터를 가져온다. 
    kernel_size=filter_size #kernal size를 설정한다. ( filter size와 같다. )
    padding_img=Get_paddig_image(img,filter_scope,filter_size) # 패딩 이미지를 가져온다. 
    w,h=img.shape[0],img.shape[1] # 이미지 사이즈 설정.
    
    #@필터링.
    for x in range(w):
        for y in range(h):# 모든 좌표 순환.
            weighted_sum = np.sum((padding_img[x:x+kernel_size,y:y+kernel_size] * filter)) #@ weighted sum elementwise multiplication
            smooth_img[x,y]=weighted_sum/np.sum(filter) #@ normalize
    return smooth_img