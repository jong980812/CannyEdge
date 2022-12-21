import argparse
import os
import cv2
import utils
from preprocessing import Convert_gray, Get_filter_size,Get_filter_scope
from gradient import Get_gradient, Gradient_Magnitude_Angle,Get_edge_LoG
from smoothing import Smoothing
from nms import Get_Discrete_angle, Get_Nms
from thresholding import Get_double_thresholding,Get_hysteresis
from utils import print_bar as bar

def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--image_path',default=None,type=str,                               help='Set img path')
    parser.add_argument('--output_img_name', type=str, default='result',                         help="set output img name")
    parser.add_argument('--run_dir', default='./', type=str,                                help='Dir python file' )
    parser.add_argument('--smoothing', default=None, choices=['gaussian', 'box'], type=str, help='Choice Smoothing method')
    parser.add_argument('--sigma', default=1.0, type=float,                                 help='Set Standard deviation')
    parser.add_argument('--th', default=0.01, type=float,                                   help='Set Threshold Value to determine filter size')
    parser.add_argument('--filter_size', default=None, type=int,                            help='Direct setting filter size, This is needed for LoG')
    parser.add_argument('--gradient_method', default='sobel', 
                       choices=['Sobel','Prewitt','LoG','DoG','Scharr'],                    help='Choose Mehotd to calculate gradient')   
    parser.add_argument('--log_th', type=float,default=0.5,                                 help='LoG threshold setting')
    parser.add_argument('--hysterisis_th',nargs='+',type=float,default=[0.1,0.2],           help="0~1 ratio")

    return parser.parse_args()

def main(args):
    img = cv2.imread(args.image_path)
    gray_img=Convert_gray(img)
    print(f"Method is {args.gradient_method}")
    output_path=os.path.join(args.run_dir,'result',args.output_img_name)
    bar()
    if args.gradient_method == "LoG":
        assert args.filter_size==5 or args.filter_size==9
        Log_mag,_=Gradient_Magnitude_Angle(gray_img, args.gradient_method, args.filter_size)
        utils.Save_img(Log_mag,output_path,'LoG_magnitude')
        Log_edge=Get_edge_LoG(Log_mag,args.log_th)
        utils.Save_img(Log_edge,output_path,'LoG_edge')
        return 
        
    if args.filter_size is None:
        filter_scope = int(Get_filter_scope(args.th, args.sigma))
        filter_size = int(Get_filter_size(filter_scope))
    else:
        filter_size=args.filter_size
        filter_scope=int((filter_size-1)/2)
    print("\n\n\n-------------START--------------------------------")
    print("Smoothing");bar()
    smoothing_img=Smoothing(gray_img, args.smoothing, args.sigma,filter_scope, filter_size)
    print("...Complete!");bar();print()
    utils.Save_img(smoothing_img,output_path,'Smoothing_img')
    

    print("Calculating Gradient");bar()
    G_mag,G_Angle=Gradient_Magnitude_Angle(gray_img, args.gradient_method, filter_size, args.sigma)
    utils.Save_img(G_mag,output_path,"Magnitude")
    utils.Save_img(G_Angle,output_path,"Angle")
    print("...Complete!");bar();print()
    
    
    print("Nonmax Surpression");bar()    
    discrete_angle=Get_Discrete_angle(G_Angle)
    nms=Get_Nms(discrete_angle, G_mag)
    utils.Save_img(nms,output_path,"Nms")
    print("...Complete!");bar();print()
    
    print("Double Thresholding");bar()
    double=Get_double_thresholding(nms,args.hysterisis_th[0],args.hysterisis_th[1])
    utils.Save_img(double,output_path,"Double_th")
    print("...Complete!");bar();print()
    
    print("Hysterisis Thresholding");bar()
    hys=Get_hysteresis(double)
    utils.Save_img(hys,output_path,"Hysterisis")
    print("...Complete!");bar()
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