import numpy as np
PI=np.pi


def MakeGrid(size : int)->np.ndarray:
    '''
    #@ size: number of grid width,height -> filter_scope
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
    scale_term = 1 / ( 2* PI * sigma**2)
    exp_term = ((x ** 2) + (y ** 2)) / (2 * (sigma ** 2))
    gaussian=scale_term * np.exp(-exp_term)
    return gaussian
    
    
def Gaussian_filter(size:int, sigma:float)->np.ndarray:
    '''
    #@ size: filter_scope
    #@ sigma: Standard deviation

    '''
    grid_x, grid_y= MakeGrid(size)                      #@ Get Grid ( like 2d coordinate )
    gaussian_filter = Get_Gaussian_2d(grid_x,grid_y,sigma)  #@ Get Gaussian 2d depend on grid size, sigma
    
    return gaussian_filter

def box_filter(size:int):
    '''
    #@ size: filter_size
    '''
    box = np.ones((size,size)) # 정의해둔 size의 행렬을 1로 채운 filter 생성 
    box_filter = box/(size**2) # 행렬 사이즈만큼 나눠서 scaling
    
    return box_filter



def Get_filter(method, sigma, filter_scope, filter_size):
    if method == "box":
        filter = box_filter(filter_size)
    elif method == "gaussian":
        filter = Gaussian_filter(filter_scope, sigma)
    print(f"Filter Scope is {filter_scope}\nFilter size is {filter_size}\nSmoothing {method} Filter \n{filter} ")
    return filter
