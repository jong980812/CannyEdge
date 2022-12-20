import numpy as np
PI=np.pi
from utils import MakeGrid,Get_paddig_image

def Roberts():
    pass
def Sobel():
    pass
def Prewitt():
    pass
def DoG(size):
    pass
def LoG(size):
    pass


def Get_Derivative_Filter(gradient:str, size=None, sigma=None)->tuple:
    '''
    #@ gradient: Filter method
    #@ size: DoG, LoG size, sigma setting
    '''
    if gradient == "DoG":
        assert size != None and sigma !=None, "DoG need kernel size and sigma"
        x,y=MakeGrid(size) #@ DoG Need Grid
        exp_term = ( x**2 + y **2 )/ (2*sigma**2)
        filter_x = x * np.exp(-exp_term)/(2*sigma*PI**4)
        filter_y = y * np.exp(-exp_term)/(2*sigma*PI**4)
        return (filter_x,filter_y)
        
def Get_gradient(img, filter):
    filter_scope = int(filter.shape[0]/2)
    filter_size = filter.shape[0]
    gradient  = np.zeros_like(img)
    padding_img = Get_paddig_image(img,filter_scope, filter_size)
    w,h=img.shape[0],img.shape[1]
    for x in range(w):
        for y in range(h):
            gradient[x,y] = np.sum((padding_img[x:x+filter_size,y:y+filter_size] * filter)) 
    return gradient

def Gradient_Magnitude_Angle(img, filter_name, size, sigma):
    '''
    #@ img: target image
    #@ filtername: gradient_filter
    #@ size: DoG, LoG size
    #@ sigma: DoG LoG sigma
    '''
    filter_x, filter_y= Get_Derivative_Filter(filter_name, size, sigma)
    gradient_x=Get_gradient(img, filter_x)
    gradient_y=Get_gradient(img, filter_y)

    magnitude=np.sqrt((gradient_x**2)+(gradient_y**2))
    magnitude = magnitude * 255 / magnitude.max()
    
    angle=np.zeros_like(gradient_x)
    angle=np.rad2deg(np.arctan2(gradient_x,gradient_y))
    return np.around(magnitude), angle+180

