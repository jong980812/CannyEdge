import numpy as np
PI=np.pi

#TODO Converting Image to Gray scale  1 channel
def Convert_gray(image : np.ndarray)->np.ndarray:
    '''
    #@ Ref: https://stackoverflow.com/questions/61058335/open-cv-how-does-the-bgr2gray-function-work
    Conver image to gray 
    Same cv2.cvtColor
    흑백으로 바꾸는 공식을 이용해서 전처리 해준다. 
    '''
    gray=np.zeros((image.shape[0],image.shape[1]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gray[i][j]+=0.299*image[i][j][2]+0.587*image[i][j][1] + 0.114*image[i][j][0]
    assert gray.shape == (image.shape[0],image.shape[1]), f"Error -> converting Failed : two ndarray's shape is different"
    
    return gray

def Get_filter_scope(threshold : float, sigma: float)->int :
    """
    -Get filter size
    -Formula
        1/(sigma*sqrt(2*pi)*exp(-x^2/(2sigma^2) = T
    -Given Gaussian N(0,sigma), User can choose threshold that determine scope used filter
    FIlter size is Odd
    Final Filter size is Scope * 2 + 1
    
    """
    ln_term = -np.log(threshold*sigma*np.sqrt(PI*2))
    scope = np.sqrt(ln_term * 2 * (sigma ** 2))
    filter_scope = np.round(scope)
    return filter_scope

def Get_filter_size(filter_scope:int)->int:
    return filter_scope * 2 + 1    