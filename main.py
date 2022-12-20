import numpy as np

def two_dim_gaussian_function(x, y, std) -> float :
    return np.exp(-1 * (x**2 + y**2) / (2 * std**2)) / (2 * np.pi * std**2)


def gaussian_filter(kernel_size, std) -> np.array(int) :  
    g_filter = np.zeros((kernel_size, kernel_size))
    half_size = kernel_size // 2
    for i, x in enumerate(range(-1 * half_size, half_size + 1)) :
        for j, y in enumerate(range(-1 * half_size, half_size + 1)) : 
            g_filter[i, j] = np.round(two_dim_gaussian_function(x, y, std))  #* g_filter[0, 0] = gaussian(-2, -2, std) if k = 3, 가우시안 필터 좌측 상단 첫번쨰 원소로는 원점에서 x축, y축 -2씩 떨어져있는 지점에서의 가우시안 값이 들어 감
    
    return g_filter

def padding(img, padding_size, value) -> np.array :
    img_size = img.shape[0]  #* img.shape = (N, N)
    padded_img = np.full((img_size + 2 * padding_size, img_size + 2 * padding_size), value)
    padded_img[padding_size : -padding_size][padding_size : -padding_size] = img
    return padded_img
     

def gaussian_smoothing(img, kernel_size, guassian_std) -> np.array :
    #* smoothing = linear filtering = convolution
    g_filter = gaussian_filter(kernel_size, guassian_std)
    
    #* to remain same img size, do padding
    #* n = input img size, f = filter size, p = one-side padding size
    #* n + 2 * p - f + 1 = n  ->  p = (f - 1) / 2
    padding_size = (kernel_size - 1) // 2  #* / -> float, // -> int
    padded_img = padding(img, padding_size, 0)
    
    img_size = img.shape[0]
    smoothed_img = np.zeros((img_size, img_size))
    
    for i in range(img_size) :
        for j in range(img_size) :
            smoothed_img[i, j] = np.sum(padded_img[i : i + kernel_size][j : j + kernel_size] * g_filter)   #* element-wise multiplication = * in numpy
            
    return smoothed_img
    
    
    
    
    
    
    