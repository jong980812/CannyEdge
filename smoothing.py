import numpy as np
from utils import Get_paddig_image
PI=np.pi


def MakeGrid(size : int)->np.ndarray:
    '''
    #@ size: number of grid width,height -> filter_scope
    #@ np.meshgrid make grid each axis( x, y )
    '''
    x,y = np.meshgrid(range(-int(size), int(size) + 1), range(-int(size), int(size) + 1))
    return x,y
def Get_Gaussian_2d(x:np.ndarray=None ,y:np.ndarray=None ,sigma=None)->np.ndarray:
    '''
    Gaussian 2d function
    Matrix calculation 2d
    #@ x: grid x
    #@ y: grid y
    '''
    scale_term = 1 / ( 2* PI * sigma**2) # gaussian coefficient
    exp_term = ((x ** 2) + (y ** 2)) / (2 * (sigma ** 2)) # exp term
    gaussian=scale_term * np.exp(-exp_term)
    return gaussian
    
    
def Gaussian_filter(size:int, sigma:float)->np.ndarray:
    '''
    #@ size: filter_scope
    #@ sigma: Standard deviation

    '''
    grid_x, grid_y= MakeGrid(size)                          #@ Get Grid ( like 2d coordinate )
    gaussian_filter = Get_Gaussian_2d(grid_x,grid_y,sigma)  #@ Get Gaussian 2d depend on grid size, sigma
    
    return gaussian_filter

def Box_filter(size:int):
    '''
    #@ size: filter_size
    '''
    box = np.ones((size,size)) # box filter have element '1' 
    box_filter = box/(size**2) # Normalize 
    return box_filter



def Get_filter(method:str, sigma:float, filter_scope:int, filter_size:int)->np.ndarray:
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
    '''
    smooth_img=np.zeros_like(img) #! Result img 
    filter=Get_filter(method, sigma, filter_scope, filter_size) #! Get filter for smoothing
    kernel_size=filter_size
    padding_img=Get_paddig_image(img,filter_scope,filter_size)
    w,h=img.shape[0],img.shape[1]
    for x in range(w):
        for y in range(h):
            weighted_sum = np.sum((padding_img[x:x+kernel_size,y:y+kernel_size] * filter)) #@ weighted sum elementwise multiplication
            smooth_img[x,y]=weighted_sum/np.sum(filter) #@ normalize
    return smooth_img