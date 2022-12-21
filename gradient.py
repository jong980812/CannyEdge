import numpy as np
PI=np.pi
from utils import MakeGrid,Get_paddig_image

    

def Get_Derivative_Filter(gradient:str, size=None, sigma=None)->tuple:
    '''
    #@ gradient: Filter method
    #@ size: DoG  size, sigma setting
    #@ FIlter setting
    '''
    if gradient == "DoG":
        assert size != None and sigma !=None, "DoG need kernel size and sigma"
        x,y=MakeGrid(size) #@ DoG Need Grid
        exp_term = ( x**2 + y **2 )/ (2*sigma**2)
        filter_x = x * np.exp(-exp_term)/(2*sigma*PI**4)
        filter_y = y * np.exp(-exp_term)/(2*sigma*PI**4)
        return (filter_x,filter_y)
    elif gradient == "Sobel":
        filter_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

        filter_y = np.array([[-1,-2,-1],
                             [ 0, 0, 0],
                             [ 1, 2, 1]])
        return (filter_x,filter_y)
    elif gradient == "Prewitt":
        filter_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])

        filter_y = np.array([[ 1, 1, 1],
                            [ 0, 0, 0],
                            [-1,-1,-1]])
        return (filter_x,filter_y)
    elif gradient == "Robert":
        filter_x = np.array([[0, 0, -1],
                             [0, 1, 0],
                             [0, 0, 0]])
        filter_y = np.array([[-1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]])
        return (filter_x,filter_y)
    elif gradient == "Scharr":
        filter_x = np.array([[-3, 0, 3],
                             [-10,0,10],
                             [-3,0,3]])
        filter_y = np.array([[-3,-10,-3],
                             [0,0,0],
                             [3,10,3]])
        return (filter_x,filter_y)
    elif gradient == "LoG":
        if size == 5:
            filter = np.array([[ 0, 0,-1, 0, 0],
                                [ 0,-1,-2,-1, 0],
                                [-1,-2,16,-2,-1],
                                [ 0,-1,-2,-1, 0],
                                [ 0, 0,-1, 0, 0]])
        elif size == 9:
            filter = np.array([[ 0, 1, 1,  2,  2,  2, 1, 1, 0],
                            [ 1, 2, 4,  5,  5,  5, 4, 2, 1],
                            [ 1, 4, 5,  3,  0,  3, 5, 4, 1],
                            [ 2, 5, 3,-12,-24,-12, 3, 5, 2],
                            [ 2, 5, 0,-24,-40,-24, 0, 5, 2],
                            [ 2, 5, 3,-12,-24,-12, 3, 5, 2],
                            [ 1, 4, 5,  3,  0,  3, 5, 4, 1],
                            [ 1, 2, 4,  5,  5,  5, 4, 2, 1],
                            [ 0, 1, 1,  2,  2,  2, 1, 1, 0]])
        return filter
        
def Get_gradient(img, filter): 
    '''
    Set Filter,
    Convolution img, Filter
    '''
    filter_scope = int(filter.shape[0]/2) #!filter scope를 수동으로 계산.
    filter_size = filter.shape[0]
    gradient  = np.zeros_like(img)#! 빈 공간 초기화.
    padding_img = Get_paddig_image(img,filter_scope, filter_size)#@ it seeme like smoothing
    w,h=img.shape[0],img.shape[1]
    for x in range(w):
        for y in range(h):
            gradient[x,y] = np.sum((padding_img[x:x+filter_size,y:y+filter_size] * filter)) #@ 필터링.컨벌루션연산과 비슷하다. 
    return gradient

def Gradient_Magnitude_Angle(img, filter_name, size, sigma=None):
    '''
    #@ img: target image
    #@ filtername: gradient_filter
    #@ size: DoG,  size
    #@ sigma: DoG  sigma
    '''
    if filter_name == "LoG":#@ LOG는 x,y구분 안해놔서 따로 구분.
        filter=Get_Derivative_Filter("LoG",size)
        gradient=Get_gradient(img,filter)
        magnitude=gradient*255/gradient.max()#! normalize
        return magnitude,None #! LoG can't return ANgle
    
    filter_x, filter_y= Get_Derivative_Filter(filter_name, size, sigma) #! Get filter x, y
    gradient_x=Get_gradient(img, filter_x)#@ 각 gradient를 받는다. 
    gradient_y=Get_gradient(img, filter_y)

    magnitude=np.sqrt((gradient_x**2)+(gradient_y**2)) #! 크기 구한다. 
    magnitude = magnitude * 255 / magnitude.max() #! normalize
    
    angle=np.zeros_like(gradient_x)
    angle=np.rad2deg(np.arctan2(gradient_y,gradient_x,)) #@ y/x to angle degree
    return np.around(magnitude), angle+180 #! 0~360으로 바꾸기위해서 
def Get_edge_LoG(mag,threshold=0.5):
    result=np.zeros_like(mag)
    threshold=100*threshold#정규화
    result[mag>=threshold] = 255 # thresholding
    return result
