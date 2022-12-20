import argparse
import os
import cv2
import utils
from preprocessing import Convert_gray, Get_filter_size,Get_filter_scope
from gradient import Get_gradient, Gradient_Magnitude_Angle
from smoothing import Smoothing
def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--image_path',default=None,type=str,                               help='Set img path')
    parser.add_argument('--run_dir', default='./', type=str,                                help='Dir python file' )
    parser.add_argument('--smoothing', default=None, choices=['gaussian', 'box'], type=str, help='Choice Smoothing method')
    parser.add_argument('--sigma', default=1.0, type=float,                                 help='Set Standard deviation')
    parser.add_argument('--th', default=0.01, type=float,                              help='Set Threshold Value to determine Kernel size')
    parser.add_argument('--kernel_size', default=None, type=int,                            help='Direct setting Kernel size')
    parser.add_argument('--gradient', default='sobel', 
                       choices=['roberts','sobel','prewitt','LoG','DoG'],                   help='Choose Mehotd to calculate gradient')    


    return parser.parse_args()

def main(args):
    img = cv2.imread(args.image_path)
    gray_img=Convert_gray(img)
    
    if args.kernel_size is None:
        filter_scope = int(Get_filter_scope(args.th, args.sigma))
        filter_size = int(Get_filter_size(filter_scope))
    else:
        filter_size=args.kernel_size
        filter_scope=int((filter_size-1)/2)

    Smoothing_img=Smoothing(gray_img, args.smoothing, args.sigma,filter_scope, filter_size)
    # utils.Show_smoothing_img(gray_img,Smoothing_img)
    G_mag,G_Angle=Gradient_Magnitude_Angle(Smoothing_img, 'DoG', filter_size, args.sigma)
    utils.Show_img(G_Angle)
    #TODO 커널과 필터 분명히 하기.
    
    
if __name__== "__main__":
    args=get_args()
    if not os.path.isdir(os.path.join(args.run_dir,'result')):  
        #@ Check output Diretory.
        '''
        If there is no 'result' directory in your run directory, 
        Make the result folder.
        '''
        output_path=os.path.join(args.run_dir,'result')
        os.mkdir(output_path)
    utils.arg_print(args)
    main(args)